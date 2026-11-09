#!/usr/bin/env python3
"""Entrenamientos principales reproducibles de CNN y modelos preentrenados."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_simple_cnn import (  # noqa: E402
    CLASS_ORDER,
    DeviceMemoryMonitor,
    SimpleCNN,
    atomic_torch_save,
    calculate_class_weights,
    capture_rng_state,
    create_augmentation_preview,
    create_unique_run_dir,
    environment_record,
    evaluate,
    is_better,
    json_dump,
    load_manifests,
    make_loader,
    plot_learning_curves,
    restore_rng_state,
    save_final_metrics,
    seed_everything,
    sha256,
    utc_now,
)


ARCHITECTURES = ("cnn_simple", "mobilenet_v3_small", "efficientnet_b0")
SEEDS = (17, 29, 43)


def validate_protocol(config: dict[str, Any], architecture: str, seed: int) -> None:
    if architecture not in ARCHITECTURES or seed not in SEEDS:
        raise RuntimeError("Arquitectura o semilla fuera del protocolo.")
    if config["data"]["class_order"] != CLASS_ORDER:
        raise RuntimeError("Orden de clases no conforme.")
    training = config["training"]
    loss_name = training["loss"]
    unweighted = loss_name == "cross_entropy_unweighted"
    if unweighted:
        if architecture != "efficientnet_b0" or config.get("secondary_comparison") != {
            "name": "efficientnet_b0_unweighted",
            "only_experimental_difference": "cross_entropy_without_class_weights",
            "preserve_primary_selection": "efficientnet_b0_weighted_seed43_epoch21",
        }:
            raise RuntimeError("La pérdida sin ponderación solo está autorizada para la comparación secundaria de EfficientNetB0.")
        if training.get("class_weight_formula") is not None:
            raise RuntimeError("La variante sin ponderación no debe definir pesos de clase.")
        if training.get("sampling") != "shuffle_without_replacement":
            raise RuntimeError("La variante secundaria debe excluir explícitamente el sobremuestreo.")
    elif loss_name != "weighted_cross_entropy":
        raise RuntimeError("Pérdida no reconocida por el protocolo.")
    expected = {
        "batch_size": 32,
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0001,
        "optimizer_reinitialized_at_fine_tuning": True,
        "automatic_mixed_precision": False,
    }
    for key, value in expected.items():
        if training[key] != value:
            raise RuntimeError(f"Configuración incompatible en training.{key}")
    if training["early_stopping"] != {
        "active_from_epoch": 6,
        "patience": 7,
        "metric": "selection_f1_macro",
    }:
        raise RuntimeError("Parada temprana incompatible con el Word.")
    if config["scope"] != {
        "allowed_partitions": ["train", "selection"],
        "calibration_used": False,
        "test_used": False,
    }:
        raise RuntimeError("El alcance permitiría consultar datos reservados.")
    phases = config["architectures"][architecture]["training_phases"]
    if architecture == "cnn_simple":
        expected_phases = [{"name": "full", "epochs": [1, 30], "extractor_learning_rate": 0.001, "classifier_learning_rate": 0.001}]
    else:
        expected_phases = [
            {"name": "frozen", "epochs": [1, 5], "extractor_learning_rate": 0.0, "classifier_learning_rate": 0.001},
            {"name": "fine_tune", "epochs": [6, 30], "extractor_learning_rate": 0.0001, "classifier_learning_rate": 0.001},
        ]
        if config["architectures"][architecture]["initialization"] != "IMAGENET1K_V1":
            raise RuntimeError("Los pesos preentrenados no son IMAGENET1K_V1.")
    if phases != expected_phases:
        raise RuntimeError("Las fases de entrenamiento no coinciden con el Word.")


def make_criterion(config: dict[str, Any], weights: np.ndarray, device: torch.device) -> nn.Module:
    if config["training"]["loss"] == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    if config["training"]["loss"] == "cross_entropy_unweighted":
        return nn.CrossEntropyLoss()
    raise RuntimeError("Pérdida no reconocida.")


def is_unweighted(config: dict[str, Any]) -> bool:
    return config["training"]["loss"] == "cross_entropy_unweighted"


def build_model(architecture: str) -> tuple[nn.Module, str]:
    if architecture == "cnn_simple":
        return SimpleCNN(), "random"
    if architecture == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, len(CLASS_ORDER))
        return model, f"{weights.__class__.__name__}.IMAGENET1K_V1"
    if architecture == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, len(CLASS_ORDER))
        return model, f"{weights.__class__.__name__}.IMAGENET1K_V1"
    raise ValueError(architecture)


def feature_parameters(model: nn.Module) -> list[nn.Parameter]:
    if not hasattr(model, "features"):
        raise RuntimeError("El modelo preentrenado no expone features.")
    return list(model.features.parameters())


def classifier_parameters(model: nn.Module) -> list[nn.Parameter]:
    if not hasattr(model, "classifier"):
        raise RuntimeError("El modelo no expone classifier.")
    return list(model.classifier.parameters())


def set_phase(model: nn.Module, architecture: str, phase: str) -> None:
    if architecture == "cnn_simple":
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
        return
    frozen = phase == "frozen"
    for parameter in feature_parameters(model):
        parameter.requires_grad = not frozen
    for parameter in classifier_parameters(model):
        parameter.requires_grad = True
    model.train()
    if frozen:
        model.features.eval()
        model.classifier.train()


def phase_for_epoch(architecture: str, epoch: int) -> str:
    if architecture == "cnn_simple":
        return "full"
    return "frozen" if epoch <= 5 else "fine_tune"


def make_optimizer(model: nn.Module, architecture: str, phase: str, config: dict[str, Any]) -> torch.optim.Optimizer:
    common = {
        "betas": tuple(config["training"]["betas"]),
        "eps": float(config["training"]["epsilon"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    if architecture == "cnn_simple":
        return torch.optim.AdamW(model.parameters(), lr=0.001, **common)
    if phase == "frozen":
        return torch.optim.AdamW(classifier_parameters(model), lr=0.001, **common)
    return torch.optim.AdamW(
        [
            {"params": feature_parameters(model), "lr": 0.0001, "group_name": "extractor"},
            {"params": classifier_parameters(model), "lr": 0.001, "group_name": "classifier"},
        ],
        **common,
    )


def learning_rates(optimizer: torch.optim.Optimizer, architecture: str, phase: str) -> tuple[float, float]:
    if architecture == "cnn_simple":
        return 0.001, 0.001
    if phase == "frozen":
        return 0.0, float(optimizer.param_groups[0]["lr"])
    return float(optimizer.param_groups[0]["lr"]), float(optimizer.param_groups[1]["lr"])


def snapshot(parameters: list[tuple[str, nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters}


def any_changed(before: dict[str, torch.Tensor], parameters: list[tuple[str, nn.Parameter]]) -> bool:
    return any(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in parameters)


def module_state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def module_state_equal(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    return all(torch.equal(before[name], value.detach().cpu()) for name, value in module.state_dict().items())


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    architecture: str,
    initialization: str,
    phase: str,
    epoch: int,
    seed: int,
    config: dict[str, Any],
    counts: np.ndarray,
    weights: np.ndarray | None,
    manifest_hashes: dict[str, str],
    history: list[dict[str, Any]],
    best: dict[str, Any],
    early_stop_counter: int,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "saved_at_utc": utc_now(),
        "run_directory": str(run_dir),
        "architecture": architecture,
        "initialization": initialization,
        "phase": phase,
        "epoch": epoch,
        "training_seed": seed,
        "partition_seed": 2026,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "class_order": CLASS_ORDER,
        "class_counts_train": counts.tolist(),
        "class_weights_train": weights.tolist() if weights is not None else None,
        "class_weights_applied": weights is not None,
        "manifest_sha256": manifest_hashes,
        "history": history,
        "best": best,
        "early_stop_counter": early_stop_counter,
        "rng_state": capture_rng_state(),
    }


def protocol_record(architecture: str, seed: int, initialization: str, config: dict[str, Any]) -> dict[str, Any]:
    transfer = architecture != "cnn_simple"
    return {
        "architecture": architecture,
        "initialization": initialization,
        "input": "RGB 224x224; resize bilineal antialias",
        "augmentation": "flip H p=0.5; flip V p=0.5; rotación uniforme 0/90/180/270",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "loss": config["training"]["loss"],
        "class_weights_applied": not is_unweighted(config),
        "class_weights_source": "solo train.csv" if not is_unweighted(config) else None,
        "sampling": config["training"].get("sampling", "shuffle_without_replacement"),
        "oversampling": False,
        "optimizer": "AdamW betas=(0.9,0.999), eps=1e-8, weight_decay=0.0001",
        "phases": (
            "épocas 1-5 clasificador lr=0.001 con extractor eval; desde 6 ajuste completo extractor lr=0.0001 y clasificador lr=0.001; optimizador reiniciado"
            if transfer
            else "entrenamiento completo lr=0.001"
        ),
        "early_stopping": "desde época 6, paciencia 7, F1 macro de selección",
        "best_checkpoint": "mayor F1 macro; empate menor pérdida no ponderada; después época anterior",
        "training_seed": seed,
        "partition_seed": 2026,
        "calibration_used": False,
        "test_used": False,
    }


def run_smoke(
    run_dir: Path,
    architecture: str,
    seed: int,
    config: dict[str, Any],
    project_root: Path,
    train: pd.DataFrame,
    selection: pd.DataFrame,
    manifest_hashes: dict[str, str],
    device: torch.device,
) -> None:
    if architecture == "cnn_simple":
        raise RuntimeError("La CNN ya dispone de prueba técnica separada.")
    seed_everything(seed)
    create_augmentation_preview(train, project_root, run_dir / "augmentation_preview.png", seed)
    seed_everything(seed)
    counts, weights = calculate_class_weights(train)
    model, initialization = build_model(architecture)
    model.to(device)
    criterion = make_criterion(config, weights, device)
    criterion_has_no_class_weights = getattr(criterion, "weight", None) is None
    if is_unweighted(config) and not criterion_has_no_class_weights:
        raise RuntimeError("La prueba técnica detectó pesos de clase en la variante sin ponderación.")
    train_loader = make_loader(train, project_root / "data", True, 32, 2, seed, 1)
    selection_loader = make_loader(selection, project_root / "data", False, 32, 2, seed, 1)
    sampler_replacement = bool(getattr(train_loader.sampler, "replacement", False))
    if sampler_replacement:
        raise RuntimeError("Se ha detectado muestreo con reemplazo no autorizado.")

    set_phase(model, architecture, "frozen")
    optimizer = make_optimizer(model, architecture, "frozen", config)
    features_before = snapshot(list(model.features.named_parameters()))
    features_state_before = module_state_snapshot(model.features)
    classifier_before = snapshot(list(model.classifier.named_parameters()))
    model.train()
    model.features.eval()
    frozen_losses: list[float] = []
    input_shape: list[int] | None = None
    for batch_index, (inputs, labels, _, _) in enumerate(train_loader):
        if batch_index >= 2:
            break
        input_shape = list(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Pérdida no finita en la fase congelada de la prueba.")
        loss.backward()
        optimizer.step()
        frozen_losses.append(float(loss.item()))
    frozen_features_unchanged = not any_changed(features_before, list(model.features.named_parameters()))
    frozen_feature_state_unchanged = module_state_equal(features_state_before, model.features)
    frozen_classifier_changed = any_changed(classifier_before, list(model.classifier.named_parameters()))
    if not frozen_features_unchanged or not frozen_feature_state_unchanged or not frozen_classifier_changed:
        raise RuntimeError("La fase congelada no respetó los parámetros previstos.")
    selection_loss, selection_metrics, predictions = evaluate(model, selection_loader, device, max_batches=2)

    set_phase(model, architecture, "fine_tune")
    optimizer = make_optimizer(model, architecture, "fine_tune", config)
    features_before_fine = snapshot(list(model.features.named_parameters()))
    classifier_before_fine = snapshot(list(model.classifier.named_parameters()))
    inputs, labels, _, _ = next(iter(make_loader(train, project_root / "data", True, 32, 2, seed, 6)))
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    fine_loss = criterion(model(inputs), labels)
    if not torch.isfinite(fine_loss):
        raise RuntimeError("Pérdida no finita en el ajuste completo de la prueba.")
    fine_loss.backward()
    optimizer.step()
    fine_features_changed = any_changed(features_before_fine, list(model.features.named_parameters()))
    fine_classifier_changed = any_changed(classifier_before_fine, list(model.classifier.named_parameters()))
    if not fine_features_changed or not fine_classifier_changed:
        raise RuntimeError("El ajuste completo no actualizó extractor y clasificador.")

    checkpoint = run_dir / "smoke_checkpoint.pt"
    atomic_torch_save({"architecture": architecture, "model_state_dict": model.state_dict(), "config": config}, checkpoint)
    model.eval()
    with torch.no_grad():
        before_reload = model(inputs)
    loaded, _ = build_model(architecture)
    loaded.to(device)
    loaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    loaded.eval()
    with torch.no_grad():
        after_reload = loaded(inputs)
    reload_exact = torch.equal(before_reload, after_reload)
    if not reload_exact:
        raise RuntimeError("La recarga del modelo no reprodujo exactamente las salidas.")
    predictions.to_csv(run_dir / "smoke_selection_predictions.csv", index=False)
    json_dump(
        run_dir / "technical_test.json",
        {
            "status": "passed",
            "architecture": architecture,
            "initialization": initialization,
            "training_seed": seed,
            "partition_seed": 2026,
            "loss": config["training"]["loss"],
            "class_weights_applied": not is_unweighted(config),
            "criterion_weight_is_none": criterion_has_no_class_weights,
            "train_sampler": type(train_loader.sampler).__name__,
            "sampling_with_replacement": sampler_replacement,
            "oversampling": False,
            "observed_input_shape": input_shape,
            "frozen_train_losses": frozen_losses,
            "frozen_features_unchanged": frozen_features_unchanged,
            "frozen_feature_parameters_and_buffers_unchanged": frozen_feature_state_unchanged,
            "frozen_classifier_changed": frozen_classifier_changed,
            "optimizer_reinitialized_for_fine_tuning": True,
            "fine_tune_loss": float(fine_loss.item()),
            "fine_tune_features_changed": fine_features_changed,
            "fine_tune_classifier_changed": fine_classifier_changed,
            "finite_selection_loss": float(selection_loss),
            "checkpoint_reload_exact": reload_exact,
            "selection_subset_metrics_for_technical_validation_only": selection_metrics,
            "manifest_sha256": manifest_hashes,
            "calibration_used": False,
            "test_used": False,
        },
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_formal(
    run_dir: Path,
    architecture: str,
    seed: int,
    config: dict[str, Any],
    project_root: Path,
    train: pd.DataFrame,
    selection: pd.DataFrame,
    manifest_hashes: dict[str, str],
    device: torch.device,
    resume: Path | None,
) -> None:
    seed_everything(seed)
    counts, weights = calculate_class_weights(train)
    model, initialization = build_model(architecture)
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = make_criterion(config, weights, device)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": None, "selection_f1_macro": float("-inf"), "selection_loss_unweighted": float("inf")}
    early_stop_counter = 0
    start_epoch = 1
    current_phase = phase_for_epoch(architecture, 1)
    set_phase(model, architecture, current_phase)
    optimizer = make_optimizer(model, architecture, current_phase, config)
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["architecture"] != architecture or state["training_seed"] != seed:
            raise RuntimeError("Arquitectura o semilla del checkpoint no coinciden.")
        if state["manifest_sha256"] != manifest_hashes or state["config"] != config:
            raise RuntimeError("Configuración o manifiestos no coinciden con el checkpoint.")
        model.load_state_dict(state["model_state_dict"])
        current_phase = state["phase"]
        set_phase(model, architecture, current_phase)
        optimizer = make_optimizer(model, architecture, current_phase, config)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best = state["best"]
        early_stop_counter = int(state["early_stop_counter"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng_state(state["rng_state"])
        previous_best = resume.parent / "best_checkpoint.pt"
        if previous_best.exists():
            shutil.copy2(previous_best, run_dir / "best_checkpoint.pt")

    applied_weights = None if is_unweighted(config) else weights
    json_dump(
        run_dir / "loss_configuration.json",
        {
            "loss": config["training"]["loss"],
            "n_train": len(train),
            "class_order": CLASS_ORDER,
            "counts": counts.tolist(),
            "class_weights_applied": applied_weights is not None,
            "weights": applied_weights.tolist() if applied_weights is not None else None,
            "sampling": config["training"].get("sampling", "shuffle_without_replacement"),
            "oversampling": False,
        },
    )
    torch.cuda.reset_peak_memory_stats()
    memory_monitor = DeviceMemoryMonitor(0)
    memory_monitor.start()
    started = time.perf_counter()
    stopped_early = False
    optimizer_restart_epochs: list[int] = []
    for epoch in range(start_epoch, 31):
        desired_phase = phase_for_epoch(architecture, epoch)
        optimizer_reinitialized = False
        if desired_phase != current_phase:
            current_phase = desired_phase
            set_phase(model, architecture, current_phase)
            optimizer = make_optimizer(model, architecture, current_phase, config)
            optimizer_reinitialized = True
            optimizer_restart_epochs.append(epoch)
        else:
            set_phase(model, architecture, current_phase)
        epoch_started = time.perf_counter()
        train_loader = make_loader(train, project_root / "data", True, 32, int(config["training"]["num_workers"]), seed, epoch)
        selection_loader = make_loader(selection, project_root / "data", False, 32, int(config["training"]["num_workers"]), seed, epoch)
        total_loss = 0.0
        total_examples = 0
        for inputs, labels, _, _ in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita en época {epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_examples += labels.size(0)
        selection_loss, metrics, _ = evaluate(model, selection_loader, device)
        torch.cuda.synchronize()
        improved = is_better(metrics, selection_loss, best)
        if improved:
            best = {"epoch": epoch, "selection_f1_macro": float(metrics["f1_macro"]), "selection_loss_unweighted": float(selection_loss)}
            if epoch >= 6:
                early_stop_counter = 0
        elif epoch >= 6:
            early_stop_counter += 1
        extractor_lr, classifier_lr = learning_rates(optimizer, architecture, current_phase)
        record = {
            "epoch": epoch,
            "phase": current_phase,
            "optimizer_reinitialized": optimizer_reinitialized,
            "extractor_learning_rate": extractor_lr,
            "classifier_learning_rate": classifier_lr,
            "train_loss": total_loss / total_examples,
            "train_loss_variant": config["training"]["loss"],
            "selection_loss_unweighted": selection_loss,
            "selection_f1_macro": metrics["f1_macro"],
            "selection_balanced_accuracy": metrics["balanced_accuracy"],
            "selection_accuracy": metrics["accuracy"],
            "improved_best": improved,
            "early_stop_counter": early_stop_counter,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "process_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "process_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }
        history.append(record)
        payload = checkpoint_payload(model, optimizer, architecture, initialization, current_phase, epoch, seed, config, counts, applied_weights, manifest_hashes, history, best, early_stop_counter, run_dir)
        atomic_torch_save(payload, run_dir / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best_checkpoint.pt")
        write_history(run_dir / "epoch_log.csv", history)
        with (run_dir / "epoch_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"architecture": architecture, "seed": seed, **record}, ensure_ascii=False), flush=True)
        if epoch >= 6 and early_stop_counter >= 7:
            stopped_early = True
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    memory_monitor.stop()
    best_state = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    final_loader = make_loader(selection, project_root / "data", False, 32, int(config["training"]["num_workers"]), seed, 0)
    best_loss, best_metrics, predictions = evaluate(model, final_loader, device)
    if abs(best_loss - float(best["selection_loss_unweighted"])) > 1e-7:
        raise RuntimeError("El checkpoint recargado no reproduce la pérdida registrada.")
    save_final_metrics(run_dir, best_metrics, predictions)
    plot_learning_curves(history, run_dir / "learning_curves.png")
    json_dump(
        run_dir / "run_summary.json",
        {
            "status": "completed",
            "architecture": architecture,
            "initialization": initialization,
            "loss": config["training"]["loss"],
            "class_weights_applied": applied_weights is not None,
            "oversampling": False,
            "training_seed": seed,
            "partition_seed": 2026,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "best_epoch": best["epoch"],
            "best_selection_loss_unweighted": best_loss,
            "best_selection_metrics": best_metrics,
            "parameter_count": parameter_count,
            "total_training_and_epoch_validation_seconds": elapsed,
            "process_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "process_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "device_peak_used_mib_nvml": memory_monitor.peak_used_mib if memory_monitor.available else None,
            "optimizer_restart_epochs": optimizer_restart_epochs,
            "manifest_sha256": manifest_hashes,
            "calibration_used": False,
            "test_used": False,
            "result_role": "selección de checkpoint; no es evaluación final",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--config", type=Path, default=Path("configs/main_training_protocol.json"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol(config, args.architecture, args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; no se entrenará en CPU.")
    train, selection, manifest_hashes = load_manifests(config, project_root)
    experiment_root = args.architecture + ("_unweighted" if is_unweighted(config) else "")
    run_dir = create_unique_run_dir(project_root / "experiments" / experiment_root, args.mode, args.seed)
    json_dump(run_dir / "config.json", {"selected_architecture": args.architecture, "selected_seed": args.seed, **config})
    json_dump(run_dir / "environment.json", environment_record())
    json_dump(run_dir / "manifest_hashes.json", manifest_hashes)
    if args.mode == "smoke":
        if args.resume:
            raise RuntimeError("La prueba técnica no admite reanudación.")
        initialization = config["architectures"][args.architecture]["initialization"]
        json_dump(run_dir / "protocol_conformance.json", protocol_record(args.architecture, args.seed, initialization, config))
        run_smoke(run_dir, args.architecture, args.seed, config, project_root, train, selection, manifest_hashes, torch.device("cuda:0"))
    else:
        initialization = config["architectures"][args.architecture]["initialization"]
        json_dump(run_dir / "protocol_conformance.json", protocol_record(args.architecture, args.seed, initialization, config))
        run_formal(run_dir, args.architecture, args.seed, config, project_root, train, selection, manifest_hashes, torch.device("cuda:0"), args.resume.resolve() if args.resume else None)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
