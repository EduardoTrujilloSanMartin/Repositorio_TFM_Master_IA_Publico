#!/usr/bin/env python3
"""Prueba técnica y entrenamiento formal de la CNN sencilla del protocolo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import pynvml
except ImportError:  # pragma: no cover - queda registrado si falta
    pynvml = None


CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def create_unique_run_dir(root: Path, kind: str, seed: int) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base = root / f"{kind}_seed{seed}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ProtocolTransform:
    def __init__(self, train: bool) -> None:
        self.train = train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = TF.resize(
            image,
            [224, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.train:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
            if torch.rand(()) < 0.5:
                image = TF.vflip(image)
            angle = int(torch.randint(0, 4, ()).item()) * 90
            if angle:
                image = TF.rotate(image, angle)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, self.mean, self.std)


class HamManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transform = ProtocolTransform(train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.frame.iloc[index]
        path = self.data_root / str(row["relative_path"])
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, CLASS_TO_INDEX[str(row["dx"])], str(row["image_id"]), str(row["lesion_id"])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 3
        for output_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.global_pool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.dropout(outputs)
        return self.classifier(outputs)


def calculate_class_weights(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = train_frame["dx"].value_counts().reindex(CLASS_ORDER).to_numpy(dtype=np.int64)
    if np.any(counts <= 0):
        raise RuntimeError(f"Alguna clase no está presente en entrenamiento: {counts.tolist()}")
    weights = len(train_frame) / (len(CLASS_ORDER) * counts.astype(np.float64))
    return counts, weights


def worker_init(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    frame: pd.DataFrame,
    data_root: Path,
    train: bool,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed * 100_000 + epoch * 101 + (1 if train else 2))
    return DataLoader(
        HamManifestDataset(frame, data_root, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=False,
    )


@dataclass
class DeviceMemoryMonitor:
    device_index: int = 0

    def __post_init__(self) -> None:
        self.peak_used_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False
        self.error = ""

    def start(self) -> None:
        if pynvml is None:
            self.error = "pynvml no está disponible"
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True

            def sample() -> None:
                while not self._stop.is_set():
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.peak_used_mib = max(self.peak_used_mib, memory.used / 1024**2)
                    self._stop.wait(0.2)

            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        except Exception as error:  # pragma: no cover - depende del controlador
            self.error = f"{type(error).__name__}: {error}"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def environment_record() -> dict[str, Any]:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        smi_output = smi.stdout.strip() if smi.returncode == 0 else smi.stderr.strip()
    except Exception as error:
        smi_output = f"{type(error).__name__}: {error}"
    return {
        "recorded_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "nvidia_smi": smi_output,
    }


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "channels": [32, 64, 128, 256],
        "dropout": 0.3,
        "num_classes": 7,
    }
    for key, value in expected.items():
        if config["architecture"][key] != value:
            raise RuntimeError(f"Configuración no conforme en architecture.{key}")
    training = config["training"]
    required = {
        "seed": 17,
        "batch_size": 32,
        "max_epochs": 30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "automatic_mixed_precision": False,
    }
    for key, value in required.items():
        if training[key] != value:
            raise RuntimeError(f"Configuración no conforme en training.{key}")
    if config["data"]["class_order"] != CLASS_ORDER:
        raise RuntimeError("El orden de clases no coincide con el protocolo.")
    if config["scope"] != {
        "allowed_partitions": ["train", "selection"],
        "calibration_used": False,
        "test_used": False,
    }:
        raise RuntimeError("El alcance permitiría usar calibración o prueba.")


def load_manifests(config: dict[str, Any], project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train_path = project_root / config["data"]["train_manifest"]
    selection_path = project_root / config["data"]["selection_manifest"]
    train = pd.read_csv(train_path)
    selection = pd.read_csv(selection_path)
    if set(train["split"].unique()) != {"train"}:
        raise RuntimeError("El manifiesto de entrenamiento contiene otras particiones.")
    if set(selection["split"].unique()) != {"selection"}:
        raise RuntimeError("El manifiesto de selección contiene otras particiones.")
    if set(train["image_id"]) & set(selection["image_id"]):
        raise RuntimeError("Hay imágenes solapadas entre entrenamiento y selección.")
    if set(train["consolidated_group_id"]) & set(selection["consolidated_group_id"]):
        raise RuntimeError("Hay grupos solapados entre entrenamiento y selección.")
    hashes = {str(train_path.relative_to(project_root)): sha256(train_path), str(selection_path.relative_to(project_root)): sha256(selection_path)}
    return train, selection, hashes


def create_augmentation_preview(train: pd.DataFrame, project_root: Path, output: Path, seed: int) -> None:
    seed_everything(seed)
    transform = ProtocolTransform(train=True)
    sample = train.sort_values("image_id").iloc[:4]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for column, (_, row) in enumerate(sample.iterrows()):
        path = project_root / "data" / row["relative_path"]
        with Image.open(path) as image:
            original = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            augmented = transform(image)
        augmented = (augmented * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, column].imshow(original)
        axes[1, column].imshow(augmented)
        axes[0, column].set_title(str(row["image_id"]))
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Determinista")
    axes[1, 0].set_ylabel("Aumentada")
    fig.suptitle("Inspección previa de transformaciones de entrenamiento")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def metric_bundle(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(CLASS_ORDER))),
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_ORDER))))
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    return {
        "f1_macro": float(f1_score(targets, predictions, labels=list(range(7)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_ORDER)
        },
        "confusion_matrix_counts": matrix.tolist(),
        "confusion_matrix_normalized_by_true_class": normalized.tolist(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    targets: list[int] = []
    predictions: list[int] = []
    rows: list[dict[str, Any]] = []
    for batch_index, (inputs, labels, image_ids, lesion_ids) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predicted = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        cpu_targets = labels.cpu().tolist()
        cpu_predictions = predicted.cpu().tolist()
        cpu_logits = logits.cpu().tolist()
        cpu_probabilities = probabilities.cpu().tolist()
        targets.extend(cpu_targets)
        predictions.extend(cpu_predictions)
        for item in range(batch_size):
            row: dict[str, Any] = {
                "image_id": image_ids[item],
                "lesion_id": lesion_ids[item],
                "true_index": cpu_targets[item],
                "true_label": CLASS_ORDER[cpu_targets[item]],
                "predicted_index": cpu_predictions[item],
                "predicted_label": CLASS_ORDER[cpu_predictions[item]],
            }
            for class_index, class_label in enumerate(CLASS_ORDER):
                row[f"logit_{class_label}"] = cpu_logits[item][class_index]
                row[f"probability_{class_label}"] = cpu_probabilities[item][class_index]
            rows.append(row)
    if total_examples == 0:
        raise RuntimeError("La evaluación no recibió imágenes.")
    return total_loss / total_examples, metric_bundle(targets, predictions), pd.DataFrame(rows)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    class_counts: np.ndarray,
    class_weights: np.ndarray,
    manifest_hashes: dict[str, str],
    history: list[dict[str, Any]],
    best: dict[str, Any],
    early_stop_counter: int,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "saved_at_utc": utc_now(),
        "run_directory": str(run_dir),
        "epoch": epoch,
        "model_class": "SimpleCNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "class_order": CLASS_ORDER,
        "class_counts_train": class_counts.tolist(),
        "class_weights_train": class_weights.tolist(),
        "manifest_sha256": manifest_hashes,
        "history": history,
        "best": best,
        "early_stop_counter": early_stop_counter,
        "rng_state": capture_rng_state(),
    }


def run_smoke(
    run_dir: Path,
    config: dict[str, Any],
    project_root: Path,
    train: pd.DataFrame,
    selection: pd.DataFrame,
    manifest_hashes: dict[str, str],
    device: torch.device,
) -> None:
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    create_augmentation_preview(train, project_root, run_dir / "augmentation_preview.png", seed)
    seed_everything(seed)
    counts, weights = calculate_class_weights(train)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0001,
    )
    loader = make_loader(train, project_root / "data", True, 32, 2, seed, 0)
    selection_loader = make_loader(selection, project_root / "data", False, 32, 2, seed, 0)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.train()
    losses: list[float] = []
    observed_shape: list[int] | None = None
    for batch_index, (inputs, labels, _, _) in enumerate(loader):
        if batch_index >= 2:
            break
        observed_shape = list(inputs.shape)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("La pérdida de la prueba técnica no es finita.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    changed = any(not torch.equal(initial[name], parameter.detach()) for name, parameter in model.named_parameters())
    if not changed:
        raise RuntimeError("La prueba técnica no actualizó los pesos.")
    selection_loss, selection_metrics, selection_predictions = evaluate(
        model, selection_loader, device, max_batches=2
    )
    if not np.isfinite(selection_loss):
        raise RuntimeError("La pérdida de selección de la prueba técnica no es finita.")
    checkpoint = run_dir / "smoke_checkpoint.pt"
    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "manifest_sha256": manifest_hashes,
        },
        checkpoint,
    )
    model.eval()
    comparison_inputs, _, _, _ = next(iter(selection_loader))
    comparison_inputs = comparison_inputs.to(device)
    with torch.no_grad():
        before = model(comparison_inputs)
    loaded = SimpleCNN().to(device)
    loaded_state = torch.load(checkpoint, map_location=device, weights_only=False)
    loaded.load_state_dict(loaded_state["model_state_dict"])
    loaded.eval()
    with torch.no_grad():
        after = loaded(comparison_inputs)
    reload_equal = torch.equal(before, after)
    if not reload_equal:
        raise RuntimeError("El modelo cargado no reproduce exactamente las salidas guardadas.")
    selection_predictions.to_csv(run_dir / "smoke_selection_predictions.csv", index=False)
    json_dump(
        run_dir / "technical_test.json",
        {
            "status": "passed",
            "completed_at_utc": utc_now(),
            "training_seed": seed,
            "partition_seed": config["data"]["partition_seed"],
            "train_batches": 2,
            "selection_batches": 2,
            "observed_input_shape": observed_shape,
            "expected_batch_input_shape": [32, 3, 224, 224],
            "finite_train_losses": losses,
            "finite_selection_loss": selection_loss,
            "weights_updated": changed,
            "checkpoint_saved": checkpoint.name,
            "checkpoint_reload_exact": reload_equal,
            "selection_subset_metrics_for_technical_validation_only": selection_metrics,
            "calibration_used": False,
            "test_used": False,
        },
    )


def is_better(metrics: dict[str, Any], selection_loss: float, best: dict[str, Any]) -> bool:
    f1 = float(metrics["f1_macro"])
    if best["epoch"] is None or f1 > best["selection_f1_macro"] + 1e-12:
        return True
    return abs(f1 - best["selection_f1_macro"]) <= 1e-12 and selection_loss < best["selection_loss_unweighted"] - 1e-12


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_learning_curves(history: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train_key = "train_loss" if "train_loss" in frame.columns else "train_loss_weighted"
    if "train_loss_variant" in frame.columns and frame["train_loss_variant"].iloc[0] == "cross_entropy_unweighted":
        train_label = "Entrenamiento sin ponderación"
    else:
        train_label = "Entrenamiento ponderado"
    axes[0].plot(frame["epoch"], frame[train_key], marker="o", label=train_label)
    axes[0].plot(frame["epoch"], frame["selection_loss_unweighted"], marker="o", label="Selección no ponderada")
    axes[0].set(xlabel="Época", ylabel="Pérdida", title="Curvas de pérdida")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(frame["epoch"], frame["selection_f1_macro"], marker="o", label="F1 macro")
    axes[1].plot(frame["epoch"], frame["selection_balanced_accuracy"], marker="o", label="Balanced accuracy")
    axes[1].set(xlabel="Época", ylabel="Métrica", title="Resultados de selección")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, output: Path, normalized: bool) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=(1 if normalized else None))
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axis.text(column, row, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
    axis.set_xticks(range(7), CLASS_ORDER, rotation=45, ha="right")
    axis.set_yticks(range(7), CLASS_ORDER)
    axis.set(xlabel="Predicción", ylabel="Clase real", title=("Matriz de confusión normalizada de selección" if normalized else "Matriz de confusión de selección"))
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_final_metrics(run_dir: Path, metrics: dict[str, Any], predictions: pd.DataFrame) -> None:
    json_dump(run_dir / "selection_metrics.json", metrics)
    pd.DataFrame(
        [{"class": label, **metrics["per_class"][label]} for label in CLASS_ORDER]
    ).to_csv(run_dir / "selection_metrics_per_class.csv", index=False)
    counts = pd.DataFrame(metrics["confusion_matrix_counts"], index=CLASS_ORDER, columns=CLASS_ORDER)
    normalized = pd.DataFrame(metrics["confusion_matrix_normalized_by_true_class"], index=CLASS_ORDER, columns=CLASS_ORDER)
    counts.to_csv(run_dir / "selection_confusion_matrix_counts.csv", index_label="true_label")
    normalized.to_csv(run_dir / "selection_confusion_matrix_normalized.csv", index_label="true_label")
    predictions.to_csv(run_dir / "selection_predictions.csv", index=False)
    plot_confusion(counts.to_numpy(), run_dir / "selection_confusion_matrix_counts.png", False)
    plot_confusion(normalized.to_numpy(), run_dir / "selection_confusion_matrix_normalized.png", True)


def run_formal(
    run_dir: Path,
    config: dict[str, Any],
    project_root: Path,
    train: pd.DataFrame,
    selection: pd.DataFrame,
    manifest_hashes: dict[str, str],
    device: torch.device,
    resume: Path | None,
) -> None:
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    counts, weights = calculate_class_weights(train)
    json_dump(
        run_dir / "class_weights_train_only.json",
        {
            "formula": training["class_weight_formula"],
            "n_train": len(train),
            "class_order": CLASS_ORDER,
            "counts": counts.tolist(),
            "weights": weights.tolist(),
        },
    )
    model = SimpleCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "epoch": None,
        "selection_f1_macro": float("-inf"),
        "selection_loss_unweighted": float("inf"),
    }
    early_stop_counter = 0
    start_epoch = 1
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["manifest_sha256"] != manifest_hashes or state["config"] != config:
            raise RuntimeError("El checkpoint de reanudación no coincide con configuración o manifiestos.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best = state["best"]
        early_stop_counter = int(state["early_stop_counter"])
        start_epoch = int(state["epoch"]) + 1
        restore_rng_state(state["rng_state"])
        previous_best = resume.parent / "best_checkpoint.pt"
        if previous_best.exists():
            shutil.copy2(previous_best, run_dir / "best_checkpoint.pt")
        elif best["epoch"] is not None:
            raise RuntimeError("Falta el best_checkpoint.pt asociado al estado de reanudación.")

    torch.cuda.reset_peak_memory_stats()
    memory_monitor = DeviceMemoryMonitor(0)
    memory_monitor.start()
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loader = make_loader(train, project_root / "data", True, 32, int(training["num_workers"]), seed, epoch)
        selection_loader = make_loader(selection, project_root / "data", False, 32, int(training["num_workers"]), seed, epoch)
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for inputs, labels, _, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita en la época {epoch}.")
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            train_loss_total += float(loss.item()) * batch_size
            train_examples += batch_size
        selection_loss, metrics, _ = evaluate(model, selection_loader, device)
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        improved = is_better(metrics, selection_loss, best)
        if improved:
            best = {
                "epoch": epoch,
                "selection_f1_macro": float(metrics["f1_macro"]),
                "selection_loss_unweighted": float(selection_loss),
            }
            if epoch >= int(training["early_stopping"]["active_from_epoch"]):
                early_stop_counter = 0
        elif epoch >= int(training["early_stopping"]["active_from_epoch"]):
            early_stop_counter += 1
        record = {
            "epoch": epoch,
            "train_loss_weighted": train_loss_total / train_examples,
            "selection_loss_unweighted": selection_loss,
            "selection_f1_macro": metrics["f1_macro"],
            "selection_balanced_accuracy": metrics["balanced_accuracy"],
            "selection_accuracy": metrics["accuracy"],
            "improved_best": improved,
            "early_stop_counter": early_stop_counter,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "process_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "process_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            config,
            counts,
            weights,
            manifest_hashes,
            history,
            best,
            early_stop_counter,
            run_dir,
        )
        atomic_torch_save(payload, run_dir / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best_checkpoint.pt")
        write_history(run_dir / "epoch_log.csv", history)
        with (run_dir / "epoch_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if epoch >= int(training["early_stopping"]["active_from_epoch"]) and early_stop_counter >= int(training["early_stopping"]["patience"]):
            stopped_early = True
            break
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    memory_monitor.stop()

    best_state = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    final_selection_loader = make_loader(selection, project_root / "data", False, 32, int(training["num_workers"]), seed, 0)
    best_loss, best_metrics, predictions = evaluate(model, final_selection_loader, device)
    if abs(best_loss - float(best["selection_loss_unweighted"])) > 1e-7:
        raise RuntimeError("La pérdida del checkpoint recargado no coincide con el registro de selección.")
    save_final_metrics(run_dir, best_metrics, predictions)
    plot_learning_curves(history, run_dir / "learning_curves.png")
    json_dump(
        run_dir / "run_summary.json",
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "run_kind": "formal",
            "training_seed": seed,
            "partition_seed": config["data"]["partition_seed"],
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "best_epoch": best["epoch"],
            "best_selection_metrics": best_metrics,
            "best_selection_loss_unweighted": best_loss,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "total_training_and_epoch_validation_seconds": total_seconds,
            "process_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "process_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "device_peak_used_mib_nvml": memory_monitor.peak_used_mib if memory_monitor.available else None,
            "device_memory_monitor_error": memory_monitor.error or None,
            "manifest_sha256": manifest_hashes,
            "calibration_used": False,
            "test_used": False,
            "result_role": "selección de checkpoint; no es evaluación final",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["smoke", "formal"])
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_simple_seed17.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/cnn_simple"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; se cancela para no ejecutar el protocolo en CPU.")
    device = torch.device("cuda:0")
    train, selection, manifest_hashes = load_manifests(config, project_root)
    run_dir = create_unique_run_dir((project_root / args.output_root).resolve(), args.mode, 17)
    json_dump(run_dir / "config.json", config)
    json_dump(run_dir / "environment.json", environment_record())
    json_dump(run_dir / "manifest_hashes.json", manifest_hashes)
    json_dump(
        run_dir / "protocol_conformance.json",
        {
            "architecture": "4 bloques Conv3x3-p1 + BatchNorm + ReLU + MaxPool2x2; 32/64/128/256; GAP; dropout 0.3; lineal 7",
            "preprocessing": "RGB; resize directo 224x224 bilineal antialias",
            "training_augmentation": "flip H p=0.5; flip V p=0.5; rotación uniforme 0/90/180/270",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "class_weights_source": "solo train.csv",
            "optimizer": "AdamW(lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0001)",
            "early_stopping": "desde época 6; paciencia 7; F1 macro de selección",
            "best_checkpoint": "mayor F1 macro; empate menor pérdida no ponderada; después época anterior",
            "training_seed": 17,
            "partition_seed": 2026,
            "calibration_used": False,
            "test_used": False,
        },
    )
    if args.mode == "smoke":
        if args.resume is not None:
            raise RuntimeError("La prueba técnica no se reanuda.")
        run_smoke(run_dir, config, project_root, train, selection, manifest_hashes, device)
    else:
        resume = args.resume.resolve() if args.resume else None
        run_formal(run_dir, config, project_root, train, selection, manifest_hashes, device, resume)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
