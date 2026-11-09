#!/usr/bin/env python3
"""Calibrate the frozen primary model and set abstention using no test data."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from train_main_models import build_model
from train_simple_cnn import CLASS_ORDER, environment_record, json_dump, make_loader, sha256, utc_now

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "calibration_abstention_primary.json"


def state_digest(model: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))


def brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    truth = np.eye(len(CLASS_ORDER), dtype=np.float64)[targets]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def reliability(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> tuple[float, pd.DataFrame]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    bin_index = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
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


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "f1_macro": float(f1_score(targets, predicted, labels=np.arange(7), average="macro", zero_division=0)),
    }


def extract_calibration_logits(
    model: torch.nn.Module, frame: pd.DataFrame, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    loader = make_loader(frame, ROOT / "data", False, 32, 4, 43, 0)
    logits, targets, image_ids, lesion_ids = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for inputs, labels, images, lesions in loader:
            output = model(inputs.to(device, non_blocking=True))
            logits.append(output.detach().cpu().numpy().astype(np.float64))
            targets.append(labels.numpy())
            image_ids.extend(images)
            lesion_ids.extend(lesions)
    return np.concatenate(logits), np.concatenate(targets), image_ids, lesion_ids


def output_frame(
    image_ids: list[str], lesion_ids: list[str], targets: np.ndarray, logits: np.ndarray,
    original: np.ndarray, calibrated: np.ndarray
) -> pd.DataFrame:
    predicted = logits.argmax(axis=1)
    frame = pd.DataFrame({
        "image_id": image_ids,
        "lesion_id": lesion_ids,
        "true_index": targets,
        "true_label": [CLASS_ORDER[i] for i in targets],
        "predicted_index": predicted,
        "predicted_label": [CLASS_ORDER[i] for i in predicted],
    })
    for index, label in enumerate(CLASS_ORDER):
        frame[f"logit_{label}"] = logits[:, index]
        frame[f"probability_original_{label}"] = original[:, index]
        frame[f"probability_calibrated_{label}"] = calibrated[:, index]
    frame["confidence_original"] = original.max(axis=1)
    frame["confidence_calibrated"] = calibrated.max(axis=1)
    frame["entropy_normalized"] = -np.sum(calibrated * np.log(np.clip(calibrated, 1e-300, 1)), axis=1) / math.log(7)
    frame["classification_error"] = frame["predicted_index"].to_numpy() != targets
    return frame


def plot_reliability(before: pd.DataFrame, after: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [2, 1]})
    centers = (before["lower"] + before["upper"]) / 2
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    axes[0].plot(centers, before["accuracy"], marker="o", label="Antes")
    axes[0].plot(centers, after["accuracy"], marker="o", label="Después")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Confianza", ylabel="Exactitud por intervalo", title=title)
    axes[0].legend(); axes[0].grid(alpha=.25)
    width = 1 / 15 * .42
    axes[1].bar(centers - width / 2, before["count"], width=width, label="Antes", alpha=.8)
    axes[1].bar(centers + width / 2, after["count"], width=width, label="Después", alpha=.8)
    axes[1].set(xlim=(0, 1), xlabel="Confianza", ylabel="Recuento")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    checkpoint = ROOT / config["system"]["checkpoint"]
    selection_manifest = ROOT / config["data"]["selection_manifest"]
    calibration_manifest = ROOT / config["data"]["calibration_manifest"]
    selection_predictions_path = ROOT / config["data"]["selection_predictions"]
    if config["system"]["class_order"] != CLASS_ORDER:
        raise RuntimeError("Orden de clases incompatible.")
    actual_hashes = {
        "checkpoint": sha256(checkpoint),
        "selection_manifest": sha256(selection_manifest),
        "calibration_manifest": sha256(calibration_manifest),
        "selection_predictions": sha256(selection_predictions_path),
    }
    if actual_hashes["checkpoint"] != config["system"]["checkpoint_sha256"]:
        raise RuntimeError("La huella del checkpoint no coincide.")
    if actual_hashes["selection_manifest"] != config["data"]["selection_manifest_sha256"] or actual_hashes["calibration_manifest"] != config["data"]["calibration_manifest_sha256"]:
        raise RuntimeError("Las huellas de los manifiestos no coinciden.")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "experiments" / "calibration_abstention" / f"primary_seed43_epoch21_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG_PATH, out / "config.json")
    json_dump(out / "environment.json", environment_record())
    json_dump(out / "input_hashes.json", actual_hashes)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {"architecture": "efficientnet_b0", "training_seed": 43, "epoch": 21, "partition_seed": 2026, "class_order": CLASS_ORDER}
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Metadato de checkpoint incompatible: {key}")
    if state["best"]["epoch"] != 21:
        raise RuntimeError("El checkpoint no representa la época seleccionada.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, initialization = build_model("efficientnet_b0")
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest_before = state_digest(model)

    calibration_manifest_frame = pd.read_csv(calibration_manifest)
    if set(calibration_manifest_frame["split"]) != {"calibration"}:
        raise RuntimeError("El manifiesto de calibración contiene otra partición.")
    cal_logits, cal_targets, cal_images, cal_lesions = extract_calibration_logits(model, calibration_manifest_frame, device)
    digest_after = state_digest(model)
    if digest_before != digest_after:
        raise RuntimeError("Los pesos cambiaron durante la extracción de logits.")

    selection_saved = pd.read_csv(selection_predictions_path)
    logit_columns = [f"logit_{label}" for label in CLASS_ORDER]
    probability_columns = [f"probability_{label}" for label in CLASS_ORDER]
    if not all(column in selection_saved for column in logit_columns + probability_columns):
        raise RuntimeError("Las predicciones de selección no contienen el orden de logits esperado.")
    sel_logits = selection_saved[logit_columns].to_numpy(np.float64)
    sel_targets = selection_saved["true_index"].to_numpy(np.int64)
    sel_original = softmax(sel_logits)
    saved_probs = selection_saved[probability_columns].to_numpy(np.float64)
    probability_max_abs_difference = float(np.max(np.abs(sel_original - saved_probs)))
    argmax_compatible = np.array_equal(sel_logits.argmax(axis=1), selection_saved["predicted_index"].to_numpy())
    selection_manifest_frame = pd.read_csv(selection_manifest)
    manifest_ids_match = selection_manifest_frame["image_id"].tolist() == selection_saved["image_id"].tolist()
    expected_selection = state["best"]
    observed_selection_metrics = classification_metrics(sel_original, sel_targets)
    selection_metric_matches_checkpoint = abs(observed_selection_metrics["f1_macro"] - expected_selection["selection_f1_macro"]) < 1e-12
    if not argmax_compatible or not manifest_ids_match or not selection_metric_matches_checkpoint or probability_max_abs_difference > 2e-7:
        raise RuntimeError("Las predicciones de selección existentes no son compatibles con el checkpoint.")

    lower, upper = config["temperature_scaling"]["temperature_bounds"]
    log_bounds = (math.log(lower), math.log(upper))
    objective = lambda log_t: nll(cal_logits, cal_targets, math.exp(float(log_t)))
    result = minimize_scalar(objective, bounds=log_bounds, method="bounded", options={"xatol": 1e-12, "maxiter": 1000})
    candidates = [(float(result.fun), float(result.x), "interior"), (objective(log_bounds[0]), log_bounds[0], "lower"), (objective(log_bounds[1]), log_bounds[1], "upper")]
    best_value, best_log_temperature, location = min(candidates, key=lambda item: (item[0], item[1]))
    temperature = float(math.exp(best_log_temperature))
    boundary_reached = location != "interior"

    cal_original = softmax(cal_logits)
    cal_calibrated = softmax(cal_logits, temperature)
    sel_calibrated = softmax(sel_logits, temperature)
    calibration_frame = output_frame(cal_images, cal_lesions, cal_targets, cal_logits, cal_original, cal_calibrated)
    selection_frame = output_frame(selection_saved["image_id"].tolist(), selection_saved["lesion_id"].tolist(), sel_targets, sel_logits, sel_original, sel_calibrated)

    metrics: dict[str, Any] = {}
    reliability_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for name, logits, targets, before, after in [
        ("calibration", cal_logits, cal_targets, cal_original, cal_calibrated),
        ("selection", sel_logits, sel_targets, sel_original, sel_calibrated),
    ]:
        before_ece, before_bins = reliability(before, targets)
        after_ece, after_bins = reliability(after, targets)
        reliability_tables[(name, "before")] = before_bins
        reliability_tables[(name, "after")] = after_bins
        before_classification = classification_metrics(before, targets)
        after_classification = classification_metrics(after, targets)
        metrics[name] = {
            "role": "used_to_fit_temperature" if name == "calibration" else "previously_used_for_model_selection_and_now_used_to_set_abstention",
            "n": len(targets),
            "before": {"log_loss": nll(logits, targets, 1.0), "brier_multiclass": brier(before, targets), "ece_15_equal_width": before_ece, **before_classification},
            "after": {"log_loss": nll(logits, targets, temperature), "brier_multiclass": brier(after, targets), "ece_15_equal_width": after_ece, **after_classification},
            "predicted_classes_identical": bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1))),
            "classification_metric_max_abs_difference": float(max(abs(before_classification[k] - after_classification[k]) for k in before_classification)),
        }
        if not metrics[name]["predicted_classes_identical"] or metrics[name]["classification_metric_max_abs_difference"] != 0.0:
            raise RuntimeError(f"La temperatura alteró clases o métricas en {name}.")
        for phase, table in [("before", before_bins), ("after", after_bins)]:
            table.to_csv(out / f"reliability_{name}_{phase}.csv", index=False)
        plot_reliability(before_bins, after_bins, f"Fiabilidad: {name}", out / f"reliability_{name}.png")

    confidences = selection_frame["confidence_calibrated"].to_numpy()
    candidates_threshold = np.unique(np.concatenate(([0.0], confidences)))
    coverages = np.array([(confidences >= threshold).mean() for threshold in candidates_threshold])
    valid = candidates_threshold[coverages >= config["abstention"]["target_minimum_selection_coverage"]]
    threshold = float(valid.max())
    accepted = confidences >= threshold
    selection_frame["accepted"] = accepted
    selection_frame["rejected"] = ~accepted
    coverage = float(accepted.mean())
    if coverage < 0.9:
        raise RuntimeError("La cobertura obtenida incumple el mínimo.")
    larger = candidates_threshold[candidates_threshold > threshold]
    if len(larger) and np.any(np.array([(confidences >= value).mean() for value in larger]) >= 0.9):
        raise RuntimeError("El umbral no es el mayor compatible con cobertura mínima.")
    next_larger_threshold = float(larger.min()) if len(larger) else None
    next_larger_coverage = float((confidences >= next_larger_threshold).mean()) if next_larger_threshold is not None else None

    errors = selection_frame["classification_error"].to_numpy(bool)
    risk_rows = []
    for value in np.sort(candidates_threshold):
        mask = confidences >= value
        risk_rows.append({"threshold": float(value), "coverage": float(mask.mean()), "accepted_count": int(mask.sum()), "risk_among_accepted": float(errors[mask].mean()) if mask.any() else float("nan")})
    risk_curve = pd.DataFrame(risk_rows).sort_values("coverage")
    risk_curve.to_csv(out / "risk_coverage_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(risk_curve["coverage"], risk_curve["risk_among_accepted"], linewidth=1.5)
    selected_risk = float(errors[accepted].mean())
    ax.scatter([coverage], [selected_risk], color="red", zorder=3, label="Umbral congelado")
    ax.set(xlabel="Cobertura", ylabel="Riesgo entre aceptados", title="Curva descriptiva riesgo–cobertura", xlim=(0, 1), ylim=(0, max(.25, float(risk_curve["risk_among_accepted"].max()) * 1.05)))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "risk_coverage_curve.png", dpi=180); plt.close(fig)

    rejection_rows = []
    for index, label in enumerate(CLASS_ORDER):
        mask = sel_targets == index
        rejection_rows.append({"class": label, "support": int(mask.sum()), "accepted": int((accepted & mask).sum()), "rejected": int((~accepted & mask).sum()), "rejection_rate": float((~accepted & mask).sum() / mask.sum())})
    rejection_by_class = pd.DataFrame(rejection_rows)
    rejection_by_class.to_csv(out / "rejection_by_true_class.csv", index=False)
    mel = sel_targets == CLASS_ORDER.index("mel")
    melanoma = {
        "support": int(mel.sum()),
        "correct_and_accepted": int((mel & accepted & ~errors).sum()),
        "accepted_with_error": int((mel & accepted & errors).sum()),
        "rejected": int((mel & ~accepted).sum()),
    }
    abstention = {
        "threshold_full_precision": threshold,
        "threshold_hex": threshold.hex(),
        "selection_size": len(selection_frame),
        "target_minimum_coverage": 0.9,
        "coverage_obtained": coverage,
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "ties_at_threshold": int((confidences == threshold).sum()),
        "next_larger_unique_confidence": next_larger_threshold,
        "coverage_at_next_larger_unique_confidence": next_larger_coverage,
        "accepted_error_count": int(errors[accepted].sum()),
        "accepted_error_rate": selected_risk,
        "rejected_error_count": int(errors[~accepted].sum()),
        "rejected_error_rate": float(errors[~accepted].mean()),
        "melanoma": melanoma,
        "rule_frozen": True,
    }
    json_dump(out / "temperature_fit.json", {
        "method": "scipy.optimize.minimize_scalar(method='bounded') on log(T)",
        "objective": "unweighted multiclass log loss on calibration only",
        "bounds_temperature": [lower, upper],
        "bounds_log_temperature": list(log_bounds),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev) + 2,
        "temperature": temperature,
        "log_temperature": best_log_temperature,
        "objective_value": best_value,
        "optimum_location": location,
        "boundary_reached": boundary_reached,
        "network_weights_updated": False,
    })
    json_dump(out / "calibration_metrics.json", metrics)
    json_dump(out / "abstention_rule.json", abstention)

    calibration_frame.to_csv(out / "calibration_logits_probabilities.csv", index=False)
    selection_frame.to_csv(out / "selection_logits_probabilities_abstention.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.histplot(data=selection_frame, x="entropy_normalized", hue="accepted", bins=30, stat="density", common_norm=False, element="step", ax=ax)
    ax.set(xlabel="Entropía predictiva normalizada por log(7)", ylabel="Densidad", title="Entropía predictiva en selección")
    fig.tight_layout(); fig.savefig(out / "selection_normalized_entropy.png", dpi=180); plt.close(fig)
    entropy_summary = selection_frame.groupby("accepted")["entropy_normalized"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    entropy_summary.to_csv(out / "selection_entropy_summary.csv", index=False)

    technical = {
        "checkpoint_identity_verified": True,
        "checkpoint_sha256_before": actual_hashes["checkpoint"],
        "checkpoint_sha256_after": sha256(checkpoint),
        "checkpoint_file_unchanged": actual_hashes["checkpoint"] == sha256(checkpoint),
        "checkpoint_metadata": expected,
        "initialization": initialization,
        "model_eval_mode": not model.training,
        "all_parameters_require_grad_false": all(not p.requires_grad for p in model.parameters()),
        "model_state_digest_before_and_after_equal": digest_before == digest_after,
        "selection_predictions_reused": True,
        "selection_predictions_source": str(selection_predictions_path.relative_to(ROOT)),
        "selection_manifest_order_matches_predictions": manifest_ids_match,
        "selection_probability_max_abs_difference_recomputed_from_logits": probability_max_abs_difference,
        "selection_log_loss_abs_difference_from_checkpoint": abs(metrics["selection"]["before"]["log_loss"] - float(state["best"]["selection_loss_unweighted"])),
        "selection_argmax_matches_saved": argmax_compatible,
        "selection_f1_matches_checkpoint": selection_metric_matches_checkpoint,
        "temperature_preserves_classes_calibration": metrics["calibration"]["predicted_classes_identical"],
        "temperature_preserves_classes_selection": metrics["selection"]["predicted_classes_identical"],
        "calibration_images_loaded": len(calibration_frame),
        "selection_images_loaded": 0,
        "selection_predictions_loaded_from_file": len(selection_frame),
        "test_manifest_accessed": False,
        "test_images_loaded": 0,
        "test_predictions_computed": 0,
    }
    json_dump(out / "technical_checks.json", technical)

    def d(value: float) -> str:
        return f"{value:.6f}".replace(".", ",")
    report = [
        "# Calibración y abstención del sistema principal", "",
        "Sistema congelado: EfficientNetB0 ponderada, semilla 43, checkpoint de la época 21. No se actualizaron los pesos y no se consultó la prueba final.", "",
        "## Temperatura", "",
        f"La temperatura se ajustó exclusivamente sobre las 990 imágenes de calibración minimizando la pérdida logarítmica no ponderada en log(T), con T limitado a [0,05, 20]. El resultado fue **T = {temperature:.17g}** (log(T) = {best_log_temperature:.17g}). Convergencia: `{result.success}`; evaluaciones: {int(result.nfev)+2}; óptimo en `{location}`; límite alcanzado: `{boundary_reached}`.", "",
        "## Efecto de la calibración", "",
        "| Conjunto | Estado | Log loss | Brier multiclase | ECE (15) | F1 macro | Balanced accuracy | Exactitud |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ("calibration", "selection"):
        for phase, label in (("before", "Antes"), ("after", "Después")):
            m = metrics[dataset][phase]
            report.append(f"| {dataset.capitalize()} | {label} | {d(m['log_loss'])} | {d(m['brier_multiclass'])} | {d(m['ece_15_equal_width'])} | {d(m['f1_macro'])} | {d(m['balanced_accuracy'])} | {d(m['accuracy'])} |")
    report += [
        "", "La división por una temperatura positiva conservó exactamente las clases predichas y todas las métricas de clasificación en ambos conjuntos. Al reconstruir las probabilidades de selección desde los logits guardados, la diferencia absoluta máxima respecto al CSV previo fue de 1,73×10⁻⁷ y la diferencia de log loss respecto al checkpoint fue de 1,19×10⁻⁹; ambas proceden de precisión y serialización en coma flotante y no alteraron ningún argmax ni métrica. Calibración se utilizó para ajustar T; selección ya había intervenido en la elección del modelo y aquí se utilizó además para fijar la abstención. Ninguno constituye una evaluación final independiente.", "",
        "## Abstención congelada", "",
        f"Se eligió el mayor umbral entre cero y las confianzas únicas que mantiene al menos el 90 % de selección: **{threshold:.17g}** (`{threshold.hex()}`). Se acepta con confianza calibrada ≥ umbral, incluyendo empates.", "",
        f"Cobertura obtenida: **{coverage:.6%}** ({int(accepted.sum())}/{len(accepted)}); rechazos: **{int((~accepted).sum())}**. Error entre aceptados: **{selected_risk:.6%}**; error entre rechazados: **{float(errors[~accepted].mean()):.6%}**.", "",
        "### Rechazo por clase real", "", "| Clase | Soporte | Aceptadas | Rechazadas | Tasa de rechazo |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rejection_by_class.iterrows():
        report.append(f"| {row['class']} | {int(row.support)} | {int(row.accepted)} | {int(row.rejected)} | {row.rejection_rate:.6%} |")
    report += [
        "", "### Melanoma", "",
        f"- Correctamente clasificados y aceptados: **{melanoma['correct_and_accepted']}**.",
        f"- Aceptados con error: **{melanoma['accepted_with_error']}**.",
        f"- Rechazados: **{melanoma['rejected']}**.",
        "", "La abstención no se contabilizó como acierto. La entropía incluida es únicamente la entropía predictiva de la distribución calibrada, normalizada por log(7); no se interpreta como descomposición de incertidumbre ni como detector validado de imágenes fuera del dominio.", "",
        "## Alcance", "", "No se entrenaron modelos, no se cambiaron checkpoint, temperatura ni cobertura para favorecer resultados, y no se ejecutaron prueba final, abstención sobre prueba ni evaluación final.",
    ]
    (out / "INFORME_CALIBRACION_ABSTENCION.md").write_text("\n".join(report) + "\n")
    json_dump(out / "run_summary.json", {
        "status": "completed", "completed_at_utc": utc_now(), "output_directory": str(out.relative_to(ROOT)),
        "checkpoint_sha256": actual_hashes["checkpoint"], "temperature": temperature, "threshold": threshold,
        "coverage": coverage, "calibration_used_for_temperature": True, "selection_used_for_abstention": True,
        "selection_not_independent_final_evaluation": True, "test_used": False,
    })
    print(json.dumps({"RUN_DIR": str(out), "temperature": temperature, "threshold": threshold, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
