#!/usr/bin/env python3
"""Evaluate the frozen primary system once on the reserved HAM10000 test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_main_models import build_model  # noqa: E402
from train_simple_cnn import (  # noqa: E402
    CLASS_ORDER,
    environment_record,
    json_dump,
    make_loader,
    seed_everything,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "final_evaluation_primary.json"
OUTPUT_ROOT = ROOT / "experiments" / "final_evaluation"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_dir(prefix: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        path = OUTPUT_ROOT / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def state_digest(model: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def log_loss_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(chosen, 1e-300, 1.0)).mean())


def brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    truth = np.eye(len(CLASS_ORDER), dtype=np.float64)[targets]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def reliability(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> tuple[float, pd.DataFrame]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    bin_index = np.minimum((confidence * bins).astype(int), bins - 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        fraction = count / len(targets)
        accuracy = float(correct[mask].mean()) if count else float("nan")
        mean_confidence = float(confidence[mask].mean()) if count else float("nan")
        gap = accuracy - mean_confidence if count else float("nan")
        contribution = fraction * abs(gap) if count else 0.0
        ece += contribution
        rows.append({
            "bin": index + 1,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "right_inclusive": index == bins - 1,
            "count": count,
            "fraction": fraction,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "accuracy_minus_confidence": gap,
            "ece_contribution": contribution,
        })
    return float(ece), pd.DataFrame(rows)


def classification_summary(targets: np.ndarray, predicted: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=np.arange(len(CLASS_ORDER)), zero_division=0
    )
    matrix = confusion_matrix(targets, predicted, labels=np.arange(len(CLASS_ORDER)))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.full(matrix.shape, np.nan, dtype=float), where=row_sums != 0)
    per_class = pd.DataFrame({
        "class": CLASS_ORDER,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    summary = {
        "n": int(len(targets)),
        "f1_macro": float(f1_score(targets, predicted, labels=np.arange(7), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "accuracy": float(accuracy_score(targets, predicted)),
    }
    return summary, per_class, matrix, normalized


def auroc_summary(targets: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    values = []
    for index, label in enumerate(CLASS_ORDER):
        binary = targets == index
        positives, negatives = int(binary.sum()), int((~binary).sum())
        if positives == 0 or negatives == 0:
            value, estimable, reason = None, False, "faltan positivos o negativos"
        else:
            value = float(roc_auc_score(binary.astype(int), probabilities[:, index]))
            estimable, reason = True, None
            values.append(value)
        rows.append({"class": label, "auroc_ovr": value, "estimable": estimable, "positives": positives, "negatives": negatives, "reason_if_not_estimable": reason})
    macro = float(np.mean(values)) if len(values) == len(CLASS_ORDER) else None
    return {
        "macro_ovr": macro,
        "macro_estimable": macro is not None,
        "macro_rule": "media de las siete AUROC; no estimable si alguna clase carece de positivos o negativos",
        "estimable_classes": len(values),
    }, pd.DataFrame(rows)


def plot_confusion(matrix: np.ndarray, normalized: bool, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    shown = matrix.astype(float)
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    threshold = np.nanmax(shown) / 2 if np.isfinite(shown).any() else 0
    for row in range(7):
        for column in range(7):
            value = shown[row, column]
            text = "NA" if not np.isfinite(value) else (f"{value:.3f}" if normalized else f"{int(value)}")
            ax.text(column, row, text, ha="center", va="center", color="white" if np.isfinite(value) and value > threshold else "black", fontsize=8)
    ax.set_xticks(range(7), CLASS_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(7), CLASS_ORDER)
    ax.set(xlabel="Clase predicha", ylabel="Clase real", title="Matriz de confusión normalizada por clase real" if normalized else "Matriz de confusión en recuentos")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_reliability(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> None:
    centers = (before["lower"] + before["upper"]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    axes[0].plot(centers, before["accuracy"], marker="o", label="Antes")
    axes[0].plot(centers, after["accuracy"], marker="o", label="Después")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Confianza", ylabel="Exactitud por intervalo", title="Fiabilidad en prueba final")
    axes[0].legend(); axes[0].grid(alpha=.25)
    width = 1 / 15 * .42
    axes[1].bar(centers - width / 2, before["count"], width=width, label="Antes", alpha=.8)
    axes[1].bar(centers + width / 2, after["count"], width=width, label="Después", alpha=.8)
    axes[1].set(xlim=(0, 1), xlabel="Confianza", ylabel="Recuento")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def materialize_group_sample(groups: dict[str, np.ndarray], sampled: list[str] | np.ndarray) -> np.ndarray:
    return np.concatenate([groups[str(group)] for group in sampled])


def aggregate_lesions(frame: pd.DataFrame, original: np.ndarray, calibrated: np.ndarray, targets: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    lesion_targets, lesion_probabilities = [], []
    for lesion_id, indices in frame.groupby("lesion_id", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        unique_targets = np.unique(targets[idx])
        if len(unique_targets) != 1:
            raise RuntimeError(f"Etiquetas contradictorias dentro de la lesión {lesion_id}.")
        original_mean = original[idx].mean(axis=0)
        calibrated_mean = calibrated[idx].mean(axis=0)
        target = int(unique_targets[0])
        predicted = int(calibrated_mean.argmax())
        row: dict[str, Any] = {
            "lesion_id": str(lesion_id), "image_count": int(len(idx)), "true_index": target,
            "true_label": CLASS_ORDER[target], "predicted_index": predicted, "predicted_label": CLASS_ORDER[predicted],
            "confidence_calibrated": float(calibrated_mean.max()),
        }
        for class_index, label in enumerate(CLASS_ORDER):
            row[f"mean_probability_original_{label}"] = float(original_mean[class_index])
            row[f"mean_probability_calibrated_{label}"] = float(calibrated_mean[class_index])
        rows.append(row); lesion_targets.append(target); lesion_probabilities.append(calibrated_mean)
    return pd.DataFrame(rows), np.asarray(lesion_targets, dtype=np.int64), np.asarray(lesion_probabilities, dtype=np.float64)


def metric_bundle(targets: np.ndarray, original: np.ndarray, calibrated: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = calibrated.argmax(axis=1)
    confidence = calibrated.max(axis=1)
    accepted = confidence >= threshold
    errors = predicted != targets
    original_ece, _ = reliability(original, targets)
    calibrated_ece, _ = reliability(calibrated, targets)
    macro_auc, _ = auroc_summary(targets, calibrated)
    all_classes_present = len(np.unique(targets)) == len(CLASS_ORDER)
    return {
        "f1_macro": float(f1_score(targets, predicted, labels=np.arange(7), average="macro", zero_division=0)) if all_classes_present else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)) if all_classes_present else float("nan"),
        "accuracy": float(accuracy_score(targets, predicted)),
        "auroc_macro_ovr": float(macro_auc["macro_ovr"]) if macro_auc["macro_ovr"] is not None else float("nan"),
        "melanoma_sensitivity": float((predicted[targets == 4] == 4).mean()) if np.any(targets == 4) else float("nan"),
        "log_loss_original": log_loss_from_probabilities(original, targets),
        "log_loss_calibrated": log_loss_from_probabilities(calibrated, targets),
        "brier_original": brier(original, targets),
        "brier_calibrated": brier(calibrated, targets),
        "ece_original": original_ece,
        "ece_calibrated": calibrated_ece,
        "coverage": float(accepted.mean()),
        "selective_risk": float(errors[accepted].mean()) if accepted.any() else float("nan"),
        "rejected_error_rate": float(errors[~accepted].mean()) if (~accepted).any() else float("nan"),
    }


def bootstrap_by_lesion(frame: pd.DataFrame, original: np.ndarray, calibrated: np.ndarray, targets: np.ndarray, threshold: float, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {str(key): np.asarray(value, dtype=int) for key, value in frame.groupby("lesion_id", sort=True).indices.items()}
    lesion_ids = np.asarray(sorted(groups), dtype=object)
    lesion_original = np.stack([original[groups[str(lesion)]].mean(axis=0) for lesion in lesion_ids])
    lesion_calibrated = np.stack([calibrated[groups[str(lesion)]].mean(axis=0) for lesion in lesion_ids])
    lesion_targets = np.asarray([targets[groups[str(lesion)]][0] for lesion in lesion_ids], dtype=np.int64)
    lesion_position = {str(lesion): index for index, lesion in enumerate(lesion_ids)}
    point_values = metric_bundle(targets, original, calibrated, threshold)
    lesion_point = metric_bundle(lesion_targets, lesion_original, lesion_calibrated, threshold)
    for metric in ("f1_macro", "balanced_accuracy", "accuracy", "auroc_macro_ovr"):
        point_values[f"lesion_{metric}"] = lesion_point[metric]
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(repetitions):
        sampled = rng.choice(lesion_ids, size=len(lesion_ids), replace=True)
        indices = materialize_group_sample(groups, sampled)
        values = metric_bundle(targets[indices], original[indices], calibrated[indices], threshold)
        sampled_positions = np.asarray([lesion_position[str(lesion)] for lesion in sampled], dtype=int)
        lesion_values = metric_bundle(lesion_targets[sampled_positions], lesion_original[sampled_positions], lesion_calibrated[sampled_positions], threshold)
        for metric in ("f1_macro", "balanced_accuracy", "accuracy", "auroc_macro_ovr"):
            values[f"lesion_{metric}"] = lesion_values[metric]
        values.update({"replicate": replicate + 1, "sampled_lesion_draws": len(sampled), "unique_lesions_in_draw": int(len(np.unique(sampled))), "image_instances": int(len(indices))})
        rows.append(values)
    replicates = pd.DataFrame(rows)
    summary_rows = []
    metric_columns = [c for c in replicates.columns if c not in {"replicate", "sampled_lesion_draws", "unique_lesions_in_draw", "image_instances"}]
    for metric in metric_columns:
        values = replicates[metric].to_numpy(float)
        valid = values[np.isfinite(values)]
        summary_rows.append({
            "metric": metric, "point_estimate": point_values[metric],
            "ci95_lower_percentile": float(np.percentile(valid, 2.5)) if len(valid) else None,
            "ci95_upper_percentile": float(np.percentile(valid, 97.5)) if len(valid) else None,
            "valid_replicates": int(len(valid)), "non_estimable_replicates": int(repetitions - len(valid)),
            "total_replicates": repetitions,
        })
    return replicates, pd.DataFrame(summary_rows)


def verify_frozen_config(config: dict[str, Any], include_test_hash: bool) -> dict[str, Any]:
    if config["system"]["class_order"] != CLASS_ORDER:
        raise RuntimeError("Orden de clases incompatible.")
    checkpoint = ROOT / config["system"]["checkpoint"]
    temperature_source = ROOT / config["frozen_calibration"]["temperature_source"]
    threshold_source = ROOT / config["frozen_calibration"]["threshold_source"]
    hashes = {
        "checkpoint": sha256(checkpoint),
        "temperature_source": sha256(temperature_source),
        "threshold_source": sha256(threshold_source),
        "evaluation_config": sha256(CONFIG_PATH),
        "evaluator": sha256(Path(__file__)),
    }
    expected = {
        "checkpoint": config["system"]["checkpoint_sha256"],
        "temperature_source": config["frozen_calibration"]["temperature_source_sha256"],
        "threshold_source": config["frozen_calibration"]["threshold_source_sha256"],
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise RuntimeError(f"Huella incompatible: {key}.")
    temperature_record = json.loads(temperature_source.read_text())
    threshold_record = json.loads(threshold_source.read_text())
    if temperature_record["temperature"] != config["frozen_calibration"]["temperature"]:
        raise RuntimeError("La temperatura completa no coincide con el archivo congelado.")
    if threshold_record["threshold_full_precision"] != config["frozen_calibration"]["threshold"]:
        raise RuntimeError("El umbral completo no coincide con el archivo congelado.")
    if not threshold_record["rule_frozen"] or temperature_record["network_weights_updated"]:
        raise RuntimeError("Los registros congelados no confirman las decisiones esperadas.")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_metadata = {"architecture": "efficientnet_b0", "training_seed": 43, "epoch": 21, "partition_seed": 2026, "class_order": CLASS_ORDER}
    for key, value in expected_metadata.items():
        if checkpoint_state.get(key) != value:
            raise RuntimeError(f"Metadato incompatible del checkpoint: {key}.")
    if include_test_hash:
        test_manifest = ROOT / config["data"]["test_manifest"]
        hashes["test_manifest"] = sha256(test_manifest)
        if hashes["test_manifest"] != config["data"]["test_manifest_sha256"]:
            raise RuntimeError("La huella del manifiesto de prueba no coincide.")
    return {"hashes": hashes, "temperature": temperature_record["temperature"], "threshold": threshold_record["threshold_full_precision"], "checkpoint_state": checkpoint_state}


def run_technical_check() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    frozen = verify_frozen_config(config, include_test_hash=False)
    selection_path = ROOT / "experiments/calibration_abstention/primary_seed43_epoch21_20261019_171549/selection_logits_probabilities_abstention.csv"
    selection = pd.read_csv(selection_path)
    logits = selection[[f"logit_{label}" for label in CLASS_ORDER]].to_numpy(np.float64)
    targets = selection["true_index"].to_numpy(np.int64)
    original = softmax(logits)
    calibrated = softmax(logits, frozen["temperature"])
    threshold = frozen["threshold"]
    selected = calibrated.max(axis=1) >= threshold
    expected_metrics = json.loads((ROOT / "experiments/calibration_abstention/primary_seed43_epoch21_20261019_171549/calibration_metrics.json").read_text())["selection"]["after"]
    observed, _, _, _ = classification_summary(targets, calibrated.argmax(axis=1))
    synthetic_logits = np.array([[4.0, 1.0, -1.0, 0.0, 2.0, 3.0, -2.0], [-2.0, 3.0, 1.0, 0.0, 2.0, -1.0, 4.0]])
    synthetic_before = softmax(synthetic_logits)
    synthetic_after = softmax(synthetic_logits, frozen["temperature"])
    equality_confidence = np.array([threshold, np.nextafter(threshold, -np.inf)])
    groups = {"a": np.array([0, 1]), "b": np.array([2])}
    multiplicity_indices = materialize_group_sample(groups, ["a", "a", "b"])
    lesion_demo = pd.DataFrame({"lesion_id": ["a", "a", "b"], "image_id": ["1", "2", "3"]})
    _, lesion_targets, lesion_probs = aggregate_lesions(lesion_demo, synthetic_before[[0, 0, 1]], synthetic_after[[0, 0, 1]], np.array([0, 0, 6]))
    checks = {
        "status": "passed",
        "completed_at_utc": utc_now(),
        "test_manifest_read": False,
        "test_images_loaded": 0,
        "frozen_hashes_verified_without_test_manifest": True,
        "checkpoint_metadata_verified": True,
        "class_order_verified": config["system"]["class_order"] == CLASS_ORDER,
        "temperature_exact_match": frozen["temperature"] == config["frozen_calibration"]["temperature"],
        "threshold_exact_match": threshold == config["frozen_calibration"]["threshold"],
        "selection_classes_match_saved": bool(np.array_equal(calibrated.argmax(axis=1), selection["predicted_index"].to_numpy())),
        "selection_acceptance_matches_saved": bool(np.array_equal(selected, selection["accepted"].to_numpy(bool))),
        "selection_f1_matches_frozen": abs(observed["f1_macro"] - expected_metrics["f1_macro"]) < 1e-12,
        "selection_balanced_accuracy_matches_frozen": abs(observed["balanced_accuracy"] - expected_metrics["balanced_accuracy"]) < 1e-12,
        "temperature_preserves_synthetic_argmax": bool(np.array_equal(synthetic_before.argmax(axis=1), synthetic_after.argmax(axis=1))),
        "threshold_equality_is_accepted": bool((equality_confidence >= threshold).tolist() == [True, False]),
        "bootstrap_group_multiplicity_preserved": bool(multiplicity_indices.tolist() == [0, 1, 0, 1, 2]),
        "lesion_aggregation_shape_correct": bool(len(lesion_targets) == 2 and lesion_probs.shape == (2, 7)),
        "probabilities_finite_and_sum_to_one": bool(np.isfinite(calibrated).all() and np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)),
        "evaluation_config_sha256": frozen["hashes"]["evaluation_config"],
        "evaluator_sha256": frozen["hashes"]["evaluator"],
        "selection_source": str(selection_path.relative_to(ROOT)),
    }
    boolean_failures = [key for key, value in checks.items() if isinstance(value, bool) and not value and key not in {"test_manifest_read"}]
    if boolean_failures:
        checks["status"] = "failed"
        raise RuntimeError(f"Falló la prueba técnica: {boolean_failures}")
    out = unique_dir("technical_check")
    shutil.copy2(CONFIG_PATH, out / "config.json")
    json_dump(out / "technical_checks.json", checks)
    print(json.dumps({"TECHNICAL_CHECK_DIR": str(out), "status": checks["status"]}, indent=2))


def find_valid_technical_check() -> Path:
    candidates = sorted(OUTPUT_ROOT.glob("technical_check_*/technical_checks.json"), reverse=True)
    if not candidates:
        raise RuntimeError("No existe una prueba técnica previa.")
    current_config = sha256(CONFIG_PATH)
    current_script = sha256(Path(__file__))
    for path in candidates:
        record = json.loads(path.read_text())
        if record.get("status") == "passed" and record.get("evaluation_config_sha256") == current_config and record.get("evaluator_sha256") == current_script and record.get("test_manifest_read") is False:
            return path
    raise RuntimeError("No existe una prueba técnica válida para esta versión de configuración y evaluador.")


def prediction_frame(manifest: pd.DataFrame, targets: np.ndarray, logits: np.ndarray, original: np.ndarray, calibrated: np.ndarray, threshold: float) -> pd.DataFrame:
    predicted = calibrated.argmax(axis=1)
    frame = manifest[["image_id", "lesion_id", "consolidated_group_id", "dx", "relative_path"]].copy()
    frame["true_index"] = targets
    frame["true_label"] = [CLASS_ORDER[index] for index in targets]
    frame["predicted_index"] = predicted
    frame["predicted_label"] = [CLASS_ORDER[index] for index in predicted]
    for index, label in enumerate(CLASS_ORDER):
        frame[f"logit_{label}"] = logits[:, index]
        frame[f"probability_original_{label}"] = original[:, index]
        frame[f"probability_calibrated_{label}"] = calibrated[:, index]
    frame["confidence_original"] = original.max(axis=1)
    frame["confidence_calibrated"] = calibrated.max(axis=1)
    frame["entropy_normalized"] = -np.sum(calibrated * np.log(np.clip(calibrated, 1e-300, 1)), axis=1) / math.log(7)
    frame["classification_correct"] = predicted == targets
    frame["classification_error"] = predicted != targets
    frame["accepted"] = frame["confidence_calibrated"] >= threshold
    frame["rejected"] = ~frame["accepted"]
    return frame


def evaluate() -> None:
    technical_path = find_valid_technical_check()
    config = json.loads(CONFIG_PATH.read_text())
    frozen = verify_frozen_config(config, include_test_hash=True)
    checkpoint = ROOT / config["system"]["checkpoint"]
    test_manifest_path = ROOT / config["data"]["test_manifest"]
    temperature, threshold = frozen["temperature"], frozen["threshold"]
    out = unique_dir("primary_seed43_epoch21")
    shutil.copy2(CONFIG_PATH, out / "config.json")
    json_dump(out / "input_hashes.json", frozen["hashes"])
    json_dump(out / "environment.json", {**environment_record(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "torchvision": torchvision.__version__})

    manifest = pd.read_csv(test_manifest_path)
    required = {"image_id", "lesion_id", "consolidated_group_id", "dx", "relative_path", "split"}
    if not required.issubset(manifest.columns) or set(manifest["split"]) != {"test"}:
        raise RuntimeError("Manifiesto de prueba incompatible.")
    if len(manifest) != 1499 or manifest["image_id"].duplicated().any():
        raise RuntimeError("Tamaño o identificadores de prueba inesperados.")
    if sorted(manifest["dx"].unique()) != sorted(CLASS_ORDER):
        raise RuntimeError("No están presentes las siete clases esperadas en prueba.")

    state = frozen["checkpoint_state"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("La evaluación formal requiere la RTX 3060 disponible mediante CUDA.")
    seed_everything(43, deterministic=True)
    model, initialization = build_model("efficientnet_b0")
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest_before = state_digest(model)
    checkpoint_hash_before = sha256(checkpoint)
    loader = make_loader(manifest, ROOT / "data", False, config["evaluation"]["batch_size"], config["evaluation"]["num_workers"], 43, 0)

    logits_parts, target_parts, image_ids, lesion_ids = [], [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for inputs, labels, images, lesions in loader:
            outputs = model(inputs.to(device, non_blocking=True))
            logits_parts.append(outputs.detach().cpu().numpy().astype(np.float64))
            target_parts.append(labels.numpy())
            image_ids.extend(images); lesion_ids.extend(lesions)
    torch.cuda.synchronize(device)
    extraction_seconds = time.perf_counter() - started
    logits = np.concatenate(logits_parts)
    targets = np.concatenate(target_parts).astype(np.int64)
    if image_ids != manifest["image_id"].astype(str).tolist() or lesion_ids != manifest["lesion_id"].astype(str).tolist():
        raise RuntimeError("La extracción no preservó el orden del manifiesto.")
    raw_logits = pd.DataFrame({"image_id": image_ids, "lesion_id": lesion_ids, "true_index": targets})
    for index, label in enumerate(CLASS_ORDER): raw_logits[f"logit_{label}"] = logits[:, index]
    raw_logits.to_csv(out / "test_logits_single_extraction.csv", index=False)
    logits_sha = sha256(out / "test_logits_single_extraction.csv")
    digest_after = state_digest(model)
    checkpoint_hash_after = sha256(checkpoint)
    extraction_record = {
        "completed_at_utc": utc_now(), "extraction_count": 1, "images": len(targets), "batches": len(logits_parts),
        "seconds": extraction_seconds, "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "logits_file": "test_logits_single_extraction.csv", "logits_file_sha256": logits_sha,
    }
    json_dump(out / "logit_extraction.json", extraction_record)

    original, calibrated = softmax(logits), softmax(logits, temperature)
    predictions = prediction_frame(manifest, targets, logits, original, calibrated, threshold)
    predictions.to_csv(out / "test_predictions_complete.csv", index=False)
    predicted = calibrated.argmax(axis=1)
    performance, per_class, matrix, matrix_normalized = classification_summary(targets, predicted)
    auroc, auroc_per_class = auroc_summary(targets, calibrated)
    performance["auroc_macro_ovr"] = auroc["macro_ovr"]
    per_class.to_csv(out / "performance_per_class.csv", index=False)
    auroc_per_class.to_csv(out / "auroc_per_class_ovr.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(out / "confusion_matrix_counts.csv")
    pd.DataFrame(matrix_normalized, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(out / "confusion_matrix_normalized_by_true_class.csv")
    plot_confusion(matrix, False, out / "confusion_matrix_counts.png")
    plot_confusion(matrix_normalized, True, out / "confusion_matrix_normalized_by_true_class.png")
    json_dump(out / "performance_metrics.json", {"image_level": performance, "auroc": auroc})

    lesion_frame, lesion_targets, lesion_probabilities = aggregate_lesions(manifest, original, calibrated, targets)
    lesion_frame.to_csv(out / "lesion_level_predictions.csv", index=False)
    lesion_performance, lesion_per_class, lesion_matrix, lesion_matrix_normalized = classification_summary(lesion_targets, lesion_probabilities.argmax(axis=1))
    lesion_auroc, lesion_auroc_per_class = auroc_summary(lesion_targets, lesion_probabilities)
    lesion_performance["auroc_macro_ovr"] = lesion_auroc["macro_ovr"]
    lesion_per_class.to_csv(out / "lesion_level_per_class.csv", index=False)
    lesion_auroc_per_class.to_csv(out / "lesion_level_auroc_per_class_ovr.csv", index=False)
    pd.DataFrame(lesion_matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(out / "lesion_level_confusion_matrix_counts.csv")
    pd.DataFrame(lesion_matrix_normalized, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(out / "lesion_level_confusion_matrix_normalized.csv")
    json_dump(out / "lesion_level_metrics.json", {"aggregation": config["evaluation"]["lesion_analysis"], "performance": lesion_performance, "auroc": lesion_auroc})

    before_ece, before_bins = reliability(original, targets)
    after_ece, after_bins = reliability(calibrated, targets)
    before_bins.to_csv(out / "reliability_before_temperature.csv", index=False)
    after_bins.to_csv(out / "reliability_after_temperature.csv", index=False)
    plot_reliability(before_bins, after_bins, out / "reliability_test.png")
    calibration = {
        "before": {"log_loss": log_loss_from_probabilities(original, targets), "brier_multiclass": brier(original, targets), "ece_15_equal_width": before_ece},
        "after": {"log_loss": log_loss_from_probabilities(calibrated, targets), "brier_multiclass": brier(calibrated, targets), "ece_15_equal_width": after_ece},
        "temperature": temperature,
        "predicted_classes_identical": bool(np.array_equal(original.argmax(axis=1), calibrated.argmax(axis=1))),
        "classification_metric_max_abs_difference": 0.0,
    }
    if not calibration["predicted_classes_identical"]:
        raise RuntimeError("La temperatura positiva alteró clases predichas.")
    json_dump(out / "calibration_metrics.json", calibration)

    accepted = predictions["accepted"].to_numpy(bool)
    errors = predictions["classification_error"].to_numpy(bool)
    rejection_rows = []
    for index, label in enumerate(CLASS_ORDER):
        mask = targets == index
        rejection_rows.append({"class": label, "support": int(mask.sum()), "accepted": int((mask & accepted).sum()), "rejected": int((mask & ~accepted).sum()), "rejection_rate": float((mask & ~accepted).sum() / mask.sum())})
    rejection_by_class = pd.DataFrame(rejection_rows)
    rejection_by_class.to_csv(out / "rejection_by_true_class.csv", index=False)
    abstention = {
        "threshold_frozen": threshold, "acceptance_rule": "confidence_calibrated >= threshold", "threshold_adapted_on_test": False,
        "n": len(targets), "coverage": float(accepted.mean()), "accepted": int(accepted.sum()), "rejected": int((~accepted).sum()),
        "accepted_errors": int(errors[accepted].sum()), "accepted_error_rate_selective_risk": float(errors[accepted].mean()) if accepted.any() else None,
        "rejected_errors": int(errors[~accepted].sum()), "rejected_error_rate": float(errors[~accepted].mean()) if (~accepted).any() else None,
    }
    json_dump(out / "abstention_metrics.json", abstention)
    unique_thresholds = np.unique(np.concatenate(([0.0], predictions["confidence_calibrated"].to_numpy(), [threshold])))
    risk_rows = []
    for value in np.sort(unique_thresholds):
        mask = predictions["confidence_calibrated"].to_numpy() >= value
        risk_rows.append({"threshold": float(value), "coverage": float(mask.mean()), "accepted_count": int(mask.sum()), "risk_among_accepted": float(errors[mask].mean()) if mask.any() else None, "is_frozen_operating_point": bool(value == threshold)})
    risk_curve = pd.DataFrame(risk_rows)
    risk_curve.to_csv(out / "risk_coverage_curve_descriptive.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    valid_risk = risk_curve.dropna(subset=["risk_among_accepted"])
    ax.plot(valid_risk["coverage"], valid_risk["risk_among_accepted"], linewidth=1.4)
    ax.scatter([abstention["coverage"]], [abstention["accepted_error_rate_selective_risk"]], color="red", label="Umbral congelado", zorder=3)
    ax.set(xlabel="Cobertura", ylabel="Riesgo entre aceptados", title="Riesgo–cobertura descriptivo en prueba final", xlim=(0, 1))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "risk_coverage_curve_descriptive.png", dpi=180); plt.close(fig)

    mel_index = CLASS_ORDER.index("mel")
    mel_true, mel_pred = targets == mel_index, predicted == mel_index
    fn_rows = []
    for index, label in enumerate(CLASS_ORDER):
        if index != mel_index: fn_rows.append({"predicted_destination": label, "count": int((mel_true & (predicted == index)).sum())})
    fp_rows = []
    for index, label in enumerate(CLASS_ORDER):
        if index != mel_index: fp_rows.append({"true_class": label, "count": int(((targets == index) & mel_pred).sum())})
    pd.DataFrame(fn_rows).to_csv(out / "melanoma_false_negative_destinations.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(out / "melanoma_false_positive_true_classes.csv", index=False)
    mel_rejected = mel_true & ~accepted
    melanoma = {
        "support": int(mel_true.sum()), "sensitivity": float((mel_true & mel_pred).sum() / mel_true.sum()),
        "true_positives": int((mel_true & mel_pred).sum()), "false_negatives": int((mel_true & ~mel_pred).sum()),
        "false_positives": int((~mel_true & mel_pred).sum()),
        "correct_and_accepted": int((mel_true & mel_pred & accepted).sum()),
        "accepted_with_error": int((mel_true & ~mel_pred & accepted).sum()),
        "rejected_total": int(mel_rejected.sum()), "rejected_correct_classification": int((mel_rejected & mel_pred).sum()),
        "rejected_incorrect_classification": int((mel_rejected & ~mel_pred).sum()), "abstention_counted_as_correct": False,
    }
    json_dump(out / "melanoma_analysis.json", melanoma)

    bootstrap_replicates, bootstrap_summary = bootstrap_by_lesion(
        manifest, original, calibrated, targets, threshold,
        config["evaluation"]["bootstrap"]["resamples"], config["evaluation"]["bootstrap"]["seed"],
    )
    bootstrap_replicates.to_csv(out / "bootstrap_lesion_replicates.csv", index=False)
    bootstrap_summary.to_csv(out / "bootstrap_lesion_ci95_percentile.csv", index=False)

    technical = {
        "technical_check_source": str(technical_path.relative_to(ROOT)), "technical_check_passed_before_test_inference": True,
        "procedural_incident": "Antes de completar la prueba técnica se mostraron cuatro filas del manifiesto al inspeccionar ruta y huella; no se cargaron imágenes ni se calcularon predicciones. La inferencia formal se realizó una sola vez después de superar la prueba técnica.",
        "checkpoint_sha256_before": checkpoint_hash_before, "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after == config["system"]["checkpoint_sha256"],
        "model_state_digest_unchanged": digest_before == digest_after, "model_eval_mode": not model.training,
        "all_parameters_require_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
        "initialization_constructor": initialization, "test_manifest_sha256_verified": frozen["hashes"]["test_manifest"] == config["data"]["test_manifest_sha256"],
        "manifest_order_preserved": image_ids == manifest["image_id"].astype(str).tolist(), "single_logit_extraction": extraction_record["extraction_count"] == 1,
        "logits_finite": bool(np.isfinite(logits).all()), "probabilities_sum_to_one": bool(np.allclose(calibrated.sum(axis=1), 1, atol=1e-12)),
        "temperature_classes_unchanged": calibration["predicted_classes_identical"], "temperature_and_threshold_loaded_from_frozen_files": True,
        "training_performed": False, "recalibration_performed": False, "threshold_adapted": False,
    }
    json_dump(out / "technical_checks.json", technical)

    ci_lookup = bootstrap_summary.set_index("metric").to_dict("index")
    report = [
        "# Evaluación final del sistema principal", "",
        "Sistema congelado: EfficientNetB0 ponderada, semilla 43, checkpoint de la época 21. La temperatura y el umbral se cargaron de sus archivos congelados. No se entrenó, recalibró ni adaptó ninguna decisión a prueba.", "",
        "## Incidencia de procedimiento", "",
        "Antes de completar la prueba técnica se mostraron accidentalmente cuatro filas del manifiesto al comprobar su ruta y huella. No se cargaron imágenes ni se generaron logits o resultados. El evaluador se congeló y superó la prueba técnica antes de la inferencia formal, que se ejecutó una sola vez. La incidencia no motivó ningún cambio de decisión.", "",
        "## Rendimiento por imagen", "",
        "| Métrica | Estimación | IC percentil 95 % por lesión | Réplicas válidas |", "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [("f1_macro", "F1 macro"), ("balanced_accuracy", "Exactitud equilibrada"), ("accuracy", "Exactitud"), ("auroc_macro_ovr", "AUROC macro OVR"), ("melanoma_sensitivity", "Sensibilidad melanoma")]:
        ci = ci_lookup[key]
        report.append(f"| {label} | {ci['point_estimate']:.6f} | [{ci['ci95_lower_percentile']:.6f}, {ci['ci95_upper_percentile']:.6f}] | {int(ci['valid_replicates'])}/1000 |")
    report += ["", "### Métricas por clase", "", "| Clase | Precisión | Recall | F1 | Soporte |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in per_class.iterrows(): report.append(f"| {row['class']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {int(row['support'])} |")
    lesion_ci_text = "; ".join(
        f"{label}: {ci_lookup[key]['point_estimate']:.6f} [{ci_lookup[key]['ci95_lower_percentile']:.6f}, {ci_lookup[key]['ci95_upper_percentile']:.6f}]"
        for key, label in [("lesion_f1_macro", "F1 macro"), ("lesion_balanced_accuracy", "exactitud equilibrada"), ("lesion_accuracy", "exactitud"), ("lesion_auroc_macro_ovr", "AUROC macro OVR")]
    )
    report += ["", "## Análisis secundario por lesión", "", f"Se promediaron las probabilidades calibradas de todas las imágenes de cada lesión antes de aplicar argmax. Hay {len(lesion_frame)} lesiones. Estimación e IC percentil del 95 %: {lesion_ci_text}.", "", "## Calibración", "", "| Estado | Log loss | Brier multiclase | ECE 15 |", "| --- | ---: | ---: | ---: |", f"| Antes | {calibration['before']['log_loss']:.6f} | {calibration['before']['brier_multiclass']:.6f} | {calibration['before']['ece_15_equal_width']:.6f} |", f"| Después | {calibration['after']['log_loss']:.6f} | {calibration['after']['brier_multiclass']:.6f} | {calibration['after']['ece_15_equal_width']:.6f} |", "", "La temperatura positiva conservó exactamente las clases predichas.", "", "## Abstención congelada", "", f"El umbral **{threshold:.17g}** se aplicó sin adaptación. Cobertura: **{abstention['coverage']:.6%}** ({abstention['accepted']}/{len(targets)}); rechazadas: **{abstention['rejected']}**; error entre aceptadas: **{abstention['accepted_error_rate_selective_risk']:.6%}**; error entre rechazadas: **{abstention['rejected_error_rate']:.6%}**.", "", "## Melanoma", "", f"Sensibilidad: **{melanoma['sensitivity']:.6f}**; falsos negativos: **{melanoma['false_negatives']}**; falsos positivos: **{melanoma['false_positives']}**. Correctos y aceptados: **{melanoma['correct_and_accepted']}**; aceptados con error: **{melanoma['accepted_with_error']}**; rechazados: **{melanoma['rejected_total']}**, de los que {melanoma['rejected_correct_classification']} eran clasificaciones correctas y {melanoma['rejected_incorrect_classification']} incorrectas. La abstención no se contabilizó como acierto.", "", "## Incertidumbre estadística", "", "Los intervalos son percentiles del 95 % de 1.000 remuestreos de lesiones con reemplazo y semilla 2027. Todas las imágenes de una lesión permanecieron juntas y las lesiones repetidas conservaron su multiplicidad. Los recuentos de réplicas válidas y no estimables se conservan en la tabla completa de bootstrap.", "", "## Separación respecto a selección", "", "Estos son resultados de prueba final obtenidos una vez con todas las decisiones congeladas. Los resultados previos de selección sirvieron para elegir arquitectura, checkpoint y umbral y no se mezclan con estas estimaciones finales.", ""]
    report += ["## Intervalos de calibración y abstención", "", "| Métrica | Estimación | IC percentil 95 % | Réplicas válidas | No estimables |", "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in [
        ("log_loss_original", "Log loss antes"), ("log_loss_calibrated", "Log loss después"),
        ("brier_original", "Brier antes"), ("brier_calibrated", "Brier después"),
        ("ece_original", "ECE antes"), ("ece_calibrated", "ECE después"),
        ("coverage", "Cobertura"), ("selective_risk", "Riesgo selectivo"),
        ("rejected_error_rate", "Error entre rechazadas"),
    ]:
        ci = ci_lookup[key]
        report.append(f"| {label} | {ci['point_estimate']:.6f} | [{ci['ci95_lower_percentile']:.6f}, {ci['ci95_upper_percentile']:.6f}] | {int(ci['valid_replicates'])} | {int(ci['non_estimable_replicates'])} |")
    report.append("")
    (out / "INFORME_EVALUACION_FINAL.md").write_text("\n".join(report), encoding="utf-8")
    json_dump(out / "run_summary.json", {
        "status": "completed", "completed_at_utc": utc_now(), "output_directory": str(out.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash_after, "test_manifest_sha256": frozen["hashes"]["test_manifest"],
        "temperature": temperature, "threshold": threshold, "logit_extraction_count": 1,
        "test_images": len(targets), "test_lesions": len(lesion_frame), "performance": performance,
        "calibration": calibration, "abstention": abstention, "melanoma": melanoma,
        "decisions_changed_after_test": False,
    })
    print(json.dumps({"RUN_DIR": str(out), "f1_macro": performance["f1_macro"], "balanced_accuracy": performance["balanced_accuracy"], "accuracy": performance["accuracy"], "coverage": abstention["coverage"], "logits_sha256": logits_sha}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--technical-check", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.technical_check:
        run_technical_check()
    else:
        evaluate()
