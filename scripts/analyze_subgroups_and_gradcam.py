#!/usr/bin/env python3
"""Exploratory subgroup analysis and deterministic Grad-CAM examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_simple_cnn import CLASS_ORDER, environment_record, json_dump, seed_everything, sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "subgroups_gradcam_primary.json"
OUTPUT_ROOT = ROOT / "experiments" / "subgroups_gradcam"
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_run_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / f"primary_test_{stamp}"
    suffix = 1
    while path.exists():
        path = OUTPUT_ROOT / f"primary_test_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def state_digest(model: nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def verify_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "test_predictions": ROOT / config["inputs"]["test_predictions"],
        "checkpoint": ROOT / config["system"]["checkpoint"],
        "final_evaluation_config": ROOT / config["inputs"]["final_evaluation_config"],
        "test_manifest": ROOT / config["inputs"]["test_manifest"],
        "original_metadata": ROOT / config["inputs"]["original_metadata"],
        "analysis_config": CONFIG_PATH,
        "analysis_script": Path(__file__),
    }
    observed = {key: sha256(path) for key, path in paths.items()}
    expected = {
        "test_predictions": config["inputs"]["test_predictions_sha256"],
        "checkpoint": config["system"]["checkpoint_sha256"],
        "final_evaluation_config": config["inputs"]["final_evaluation_config_sha256"],
        "test_manifest": config["inputs"]["test_manifest_sha256"],
        "original_metadata": config["inputs"]["original_metadata_sha256"],
    }
    for key, expected_hash in expected.items():
        if observed[key] != expected_hash:
            raise RuntimeError(f"Huella incompatible para {key}: {observed[key]} != {expected_hash}")
    if config["system"]["class_order"] != CLASS_ORDER:
        raise RuntimeError("Orden de clases incompatible.")
    final_config = json.loads(paths["final_evaluation_config"].read_text())
    if final_config["system"]["checkpoint_sha256"] != config["system"]["checkpoint_sha256"]:
        raise RuntimeError("El checkpoint no coincide con la evaluación final.")
    if final_config["frozen_calibration"]["temperature"] != config["system"]["temperature"]:
        raise RuntimeError("La temperatura no coincide con la evaluación final.")
    if final_config["frozen_calibration"]["threshold"] != config["system"]["threshold"]:
        raise RuntimeError("El umbral no coincide con la evaluación final.")
    return observed


def merge_predictions_metadata(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = pd.read_csv(ROOT / config["inputs"]["test_predictions"])
    metadata = pd.read_csv(ROOT / config["inputs"]["original_metadata"])
    if len(predictions) != predictions["image_id"].nunique() or len(metadata) != metadata["image_id"].nunique():
        raise RuntimeError("Hay image_id duplicados en predicciones o metadatos.")
    merged = predictions.merge(metadata, on="image_id", how="left", validate="one_to_one", suffixes=("", "_metadata"), indicator=True)
    if len(merged) != len(predictions) or not (merged["_merge"] == "both").all():
        raise RuntimeError("La unión uno a uno perdió o añadió registros.")
    if not (merged["dx"] == merged["dx_metadata"]).all():
        raise RuntimeError("Las etiquetas de predicción y metadatos no coinciden.")
    if not (merged["lesion_id"] == merged["lesion_id_metadata"]).all():
        raise RuntimeError("Los identificadores de lesión no coinciden.")
    if not (merged["true_label"] == merged["dx"]).all():
        raise RuntimeError("La referencia codificada no coincide con dx.")
    merged = merged.drop(columns=["_merge", "lesion_id_metadata", "dx_metadata"])
    merged["age_group"] = "unknown"
    observed_age = merged["age"].notna()
    merged.loc[observed_age & (merged["age"] < 40), "age_group"] = "<40"
    merged.loc[observed_age & (merged["age"] >= 40) & (merged["age"] <= 59), "age_group"] = "40-59"
    merged.loc[observed_age & (merged["age"] >= 60), "age_group"] = ">=60"
    for column in ("sex", "localization"):
        merged[column] = merged[column].fillna("unknown").astype(str).str.strip().replace({"": "unknown"})
    checks = {
        "prediction_rows": int(len(predictions)), "metadata_rows": int(len(metadata)), "joined_rows": int(len(merged)),
        "prediction_image_ids_unique": True, "metadata_image_ids_unique": True,
        "one_to_one_join": True, "unmatched_predictions": 0, "duplicated_after_join": int(merged["image_id"].duplicated().sum()),
        "label_mismatches": 0, "lesion_id_mismatches": 0,
        "missing_age_kept_as_unknown": int(merged["age"].isna().sum()),
        "missing_or_explicit_unknown_sex": int((merged["sex"] == "unknown").sum()),
        "missing_or_explicit_unknown_localization": int((merged["localization"] == "unknown").sum()),
    }
    return merged, checks


def subgroup_tables(merged: pd.DataFrame, variable: str, threshold_lesions: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    values = sorted(merged[variable].astype(str).unique(), key=lambda value: (value == "unknown", value))
    if variable == "age_group":
        values = [value for value in ["<40", "40-59", ">=60", "unknown"] if value in values]
    for value in values:
        group = merged[merged[variable].astype(str) == value]
        targets = group["true_index"].to_numpy(int)
        predicted = group["predicted_index"].to_numpy(int)
        all_classes = set(targets.tolist()) == set(range(7))
        accepted = group["accepted"].to_numpy(bool)
        errors = group["classification_error"].to_numpy(bool)
        lesions = int(group["lesion_id"].nunique())
        summary_rows.append({
            "variable": variable, "group": value, "images": int(len(group)), "lesions": lesions,
            "scarce_evidence_less_than_20_lesions": lesions < threshold_lesions,
            "all_seven_true_classes_present": all_classes,
            "accuracy": float(accuracy_score(targets, predicted)),
            "f1_macro": float(f1_score(targets, predicted, labels=np.arange(7), average="macro", zero_division=0)) if all_classes else None,
            "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)) if all_classes else None,
            "macro_metrics_estimable_comparably": all_classes,
            "coverage": float(accepted.mean()), "rejection_rate": float((~accepted).mean()),
            "accepted_count": int(accepted.sum()), "rejected_count": int((~accepted).sum()),
            "accepted_error_rate": float(errors[accepted].mean()) if accepted.any() else None,
            "melanoma_support": int((targets == CLASS_TO_INDEX["mel"]).sum()),
            "melanoma_sensitivity": float((predicted[targets == CLASS_TO_INDEX["mel"]] == CLASS_TO_INDEX["mel"]).mean()) if np.any(targets == CLASS_TO_INDEX["mel"]) else None,
        })
        for class_index, label in enumerate(CLASS_ORDER):
            true_mask = targets == class_index
            predicted_mask = predicted == class_index
            support = int(true_mask.sum())
            predicted_positive = int(predicted_mask.sum())
            tp = int((true_mask & predicted_mask).sum())
            if support:
                precision = tp / predicted_positive if predicted_positive else 0.0
                recall = tp / support
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                estimable, reason = True, ""
            else:
                precision = recall = f1 = None
                estimable, reason = False, "no hay casos reales de la clase en el grupo"
            composition_rows.append({
                "variable": variable, "group": value, "class": label, "images": support,
                "proportion_within_group": support / len(group),
                "lesions": int(group.loc[true_mask, "lesion_id"].nunique()),
            })
            class_rows.append({
                "variable": variable, "group": value, "class": label, "support": support,
                "predicted_positive": predicted_positive, "true_positive": tp,
                "precision": precision, "recall": recall, "f1": f1,
                "metrics_estimable": estimable, "reason_if_not_estimable": reason,
                "group_lesions": lesions, "scarce_group": lesions < threshold_lesions,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(composition_rows), pd.DataFrame(class_rows)


def plot_subgroups(summary: pd.DataFrame, composition: pd.DataFrame, variable: str, out: Path) -> None:
    order = summary["group"].tolist()
    width = max(8, 0.72 * len(order) + 4)
    fig, ax = plt.subplots(figsize=(width, 5.2))
    x = np.arange(len(order)); bar_width = .25
    ax.bar(x - bar_width, summary["accuracy"], bar_width, label="Exactitud")
    ax.bar(x, summary["coverage"], bar_width, label="Cobertura")
    ax.bar(x + bar_width, summary["rejection_rate"], bar_width, label="Rechazo")
    ax.set_xticks(x, order, rotation=40 if len(order) > 6 else 0, ha="right" if len(order) > 6 else "center")
    ax.set_ylim(0, 1); ax.set_ylabel("Proporción"); ax.set_title(f"Resultados exploratorios por {variable}")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / f"subgroups_{variable}_performance.png", dpi=180); plt.close(fig)
    pivot = composition.pivot(index="group", columns="class", values="proportion_within_group").reindex(index=order, columns=CLASS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(width, 5.2)); bottom = np.zeros(len(pivot))
    for label in CLASS_ORDER:
        values = pivot[label].to_numpy(); ax.bar(np.arange(len(pivot)), values, bottom=bottom, label=label); bottom += values
    ax.set_xticks(np.arange(len(pivot)), pivot.index, rotation=40 if len(order) > 6 else 0, ha="right" if len(order) > 6 else "center")
    ax.set_ylim(0, 1); ax.set_ylabel("Proporción de imágenes"); ax.set_title(f"Composición diagnóstica por {variable}")
    ax.legend(ncol=4, fontsize=8); fig.tight_layout(); fig.savefig(out / f"subgroups_{variable}_composition.png", dpi=180); plt.close(fig)


def within_class_ranges(class_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variable, label), group in class_metrics.groupby(["variable", "class"], sort=False):
        estimable = group[group["metrics_estimable"] & ~group["scarce_group"]]
        recalls = estimable["recall"].dropna()
        rows.append({
            "variable": variable, "class": label, "non_scarce_estimable_groups": int(len(recalls)),
            "minimum_recall": float(recalls.min()) if len(recalls) else None,
            "maximum_recall": float(recalls.max()) if len(recalls) else None,
            "recall_range": float(recalls.max() - recalls.min()) if len(recalls) >= 2 else None,
            "interpretation": "descriptivo; no causal y condicionado por composición y tamaño muestral",
        })
    return pd.DataFrame(rows)


def choose_examples(merged: pd.DataFrame, maximum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows, availability_rows = [], []
    for true_label in CLASS_ORDER:
        class_frame = merged[merged["true_label"] == true_label].sort_values("image_id")
        used_images: set[str] = set(); used_lesions: set[str] = set()
        category_masks = {
            "correct_accepted": class_frame["accepted"] & class_frame["classification_correct"],
            "error_accepted": class_frame["accepted"] & class_frame["classification_error"],
            "rejected": class_frame["rejected"],
        }
        for category in ("rejected", "correct_accepted", "error_accepted"):
            candidates = class_frame[category_masks[category]].sort_values("image_id")
            chosen_indices: list[int] = []
            for index, row in candidates.iterrows():
                if len(chosen_indices) >= maximum: break
                if str(row["image_id"]) not in used_images and str(row["lesion_id"]) not in used_lesions:
                    chosen_indices.append(index); used_images.add(str(row["image_id"])); used_lesions.add(str(row["lesion_id"]))
            if len(chosen_indices) < maximum:
                for index, row in candidates.iterrows():
                    if len(chosen_indices) >= maximum: break
                    if index not in chosen_indices and str(row["image_id"]) not in used_images:
                        chosen_indices.append(index); used_images.add(str(row["image_id"])); used_lesions.add(str(row["lesion_id"]))
            chosen = candidates.loc[chosen_indices].copy()
            for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                selected_rows.append({
                    "true_class": true_label, "selection_category": category, "rank": rank,
                    "image_id": row["image_id"], "lesion_id": row["lesion_id"], "relative_path": row["relative_path"],
                    "predicted_class": row["predicted_label"], "predicted_index": int(row["predicted_index"]),
                    "true_index": int(row["true_index"]), "confidence_calibrated": float(row["confidence_calibrated"]),
                    "accepted": bool(row["accepted"]), "rejected": bool(row["rejected"]),
                    "classification_correct": bool(row["classification_correct"]),
                })
            availability_rows.append({
                "true_class": true_label, "selection_category": category,
                "available_candidates": int(len(candidates)), "requested_maximum": maximum,
                "selected": int(len(chosen)), "missing_to_maximum": int(maximum - len(chosen)),
            })
    selected = pd.DataFrame(selected_rows).sort_values(["true_class", "selection_category", "rank"]).reset_index(drop=True)
    if selected["image_id"].duplicated().any():
        raise RuntimeError("La selección contiene imágenes repetidas.")
    return selected, pd.DataFrame(availability_rows)


def prepare() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    hashes = verify_inputs(config)
    out = unique_run_dir()
    shutil.copy2(CONFIG_PATH, out / "config.json")
    json_dump(out / "input_hashes.json", hashes)
    merged, join_checks = merge_predictions_metadata(config)
    merged.to_csv(out / "test_predictions_with_original_metadata.csv", index=False)
    all_summary, all_composition, all_class = [], [], []
    for variable in ("age_group", "sex", "localization"):
        summary, composition, class_metrics = subgroup_tables(merged, variable, config["subgroups"]["scarce_evidence_lesions_below"])
        all_summary.append(summary); all_composition.append(composition); all_class.append(class_metrics)
        plot_subgroups(summary, composition, variable, out)
    summary = pd.concat(all_summary, ignore_index=True)
    composition = pd.concat(all_composition, ignore_index=True)
    class_metrics = pd.concat(all_class, ignore_index=True)
    summary.to_csv(out / "subgroup_summary.csv", index=False)
    composition.to_csv(out / "subgroup_class_composition.csv", index=False)
    class_metrics.to_csv(out / "subgroup_metrics_per_class.csv", index=False)
    within_class_ranges(class_metrics).to_csv(out / "within_class_recall_ranges_non_scarce.csv", index=False)
    selected, availability = choose_examples(merged, config["example_selection"]["maximum_per_true_class_and_category"])
    selected.to_csv(out / "selected_examples.csv", index=False)
    availability.to_csv(out / "selected_examples_availability.csv", index=False)
    observations = selected[["image_id", "lesion_id", "true_class", "predicted_class", "selection_category"]].copy()
    for column in ("hair_visible", "ruler_visible", "mark_visible", "border_visible"):
        observations[column] = "pending_visual_review"
    observations["notes"] = "No se realizó interpretación automática; completar mediante revisión visual descriptiva."
    observations.to_csv(out / "visual_observation_checklist.csv", index=False)
    technical = {
        **join_checks, "input_hashes_verified": True, "class_order_verified": config["system"]["class_order"] == CLASS_ORDER,
        "statistics_source_only_saved_predictions": True, "full_test_inference_repeated": False,
        "selection_rule_frozen_before_gradcam": True, "selection_rule": config["example_selection"],
        "selected_images": int(len(selected)), "selected_image_ids_unique": not selected["image_id"].duplicated().any(),
        "selected_examples_sha256": sha256(out / "selected_examples.csv"), "gradcam_generated": False,
    }
    json_dump(out / "technical_checks_prepare.json", technical)
    json_dump(out / "prepare_summary.json", {
        "status": "prepared", "prepared_at_utc": utc_now(), "output_directory": str(out.relative_to(ROOT)),
        "analysis_config_sha256": sha256(CONFIG_PATH), "analysis_script_sha256": sha256(Path(__file__)),
        "selected_examples_sha256": sha256(out / "selected_examples.csv"), "selected_images": len(selected),
        "subgroup_count": len(summary), "scarce_groups": int(summary["scarce_evidence_less_than_20_lesions"].sum()),
    })
    print(json.dumps({"RUN_DIR": str(out), "selected_images": len(selected), "scarce_groups": int(summary["scarce_evidence_less_than_20_lesions"].sum())}, indent=2))


def build_frozen_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    model = torchvision.models.efficientnet_b0(weights=None)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, len(CLASS_ORDER))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {"architecture": "efficientnet_b0", "training_seed": 43, "epoch": 21, "partition_seed": 2026, "class_order": CLASS_ORDER}
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Metadato de checkpoint incompatible: {key}")
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model, state


def load_image_tensors(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        resized = TF.resize(rgb, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        display = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = TF.normalize(TF.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    return tensor, display


def gradcam_map(logit: torch.Tensor, activation: torch.Tensor, retain_graph: bool) -> tuple[np.ndarray, dict[str, Any]]:
    gradient = torch.autograd.grad(logit, activation, retain_graph=retain_graph, create_graph=False)[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    raw = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    raw = torch.nn.functional.interpolate(raw, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    finite = bool(np.isfinite(raw).all())
    minimum = float(np.nanmin(raw)) if finite else None
    maximum = float(np.nanmax(raw)) if finite else None
    span = maximum - minimum if finite else None
    null_map = (not finite) or span is None or span <= 1e-12 or maximum <= 1e-12
    normalized = np.zeros_like(raw, dtype=np.float32) if null_map else ((raw - minimum) / span).astype(np.float32)
    return normalized, {"finite": finite, "raw_minimum": minimum, "raw_maximum": maximum, "raw_span": span, "null_map": bool(null_map)}


def overlay(display: np.ndarray, cam: np.ndarray, opacity: float) -> np.ndarray:
    colored = plt.get_cmap("jet")(cam)[..., :3].astype(np.float32)
    return np.clip((1 - opacity) * display + opacity * colored, 0, 1)


def save_individual_panel(display: np.ndarray, predicted_overlay: np.ndarray, true_overlay: np.ndarray | None, row: pd.Series, path: Path) -> None:
    columns = 3 if true_overlay is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(5.0 * columns, 5.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(display); axes[0].set_title("Imagen redimensionada")
    axes[1].imshow(predicted_overlay); axes[1].set_title(f"Grad-CAM predicha: {row['predicted_class']}")
    if true_overlay is not None:
        axes[2].imshow(true_overlay); axes[2].set_title(f"Grad-CAM referencia: {row['true_class']}")
    for ax in axes: ax.axis("off")
    state = "aceptada" if bool(row["accepted"]) else "rechazada"
    fig.suptitle(f"{row['image_id']} · lesión {row['lesion_id']}\nreal {row['true_class']} · predicha {row['predicted_class']} · confianza {row['confidence_calibrated']:.4f} · {state}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def generate_gradcam(run_dir: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    hashes = verify_inputs(config)
    prepare_summary = json.loads((run_dir / "prepare_summary.json").read_text())
    if prepare_summary["analysis_config_sha256"] != sha256(CONFIG_PATH) or prepare_summary["analysis_script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("La configuración o el script cambiaron después de congelar la selección.")
    selected_path = run_dir / "selected_examples.csv"
    if sha256(selected_path) != prepare_summary["selected_examples_sha256"]:
        raise RuntimeError("La lista de ejemplos cambió después de su selección.")
    selected = pd.read_csv(selected_path)
    predictions = pd.read_csv(ROOT / config["inputs"]["test_predictions"]).set_index("image_id")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Grad-CAM se ejecutará en la RTX 3060; CUDA no está disponible.")
    seed_everything(43, deterministic=True)
    checkpoint = ROOT / config["system"]["checkpoint"]
    checkpoint_before = sha256(checkpoint)
    model, _ = build_frozen_model(checkpoint, device)
    digest_before = state_digest(model)
    target_layer = dict(model.named_modules()).get(config["gradcam"]["target_layer"])
    if target_layer is None or not isinstance(target_layer, nn.Conv2d):
        raise RuntimeError("La capa Grad-CAM configurada no es una Conv2d válida.")
    activations: list[torch.Tensor] = []
    hook = target_layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    originals_dir = run_dir / "selected_originals"; maps_dir = run_dir / "gradcam_maps"; panels_dir = run_dir / "individual_panels"; class_panels_dir = run_dir / "class_panels"
    for directory in (originals_dir, maps_dir, panels_dir, class_panels_dir): directory.mkdir(exist_ok=False)
    map_rows, example_rows = [], []
    panel_overlays: dict[str, np.ndarray] = {}
    maximum_probability_difference = 0.0
    try:
        for _, row in selected.iterrows():
            image_id = str(row["image_id"]); source = ROOT / "data" / str(row["relative_path"])
            shutil.copy2(source, originals_dir / source.name)
            tensor_single, display = load_image_tensors(source)
            # La evaluación final usó lote 32. En CUDA, la forma del lote cambia
            # ligeramente la aritmética convolucional; repetir el mismo ejemplo
            # reproduce esa forma sin cargar imágenes de prueba no seleccionadas.
            tensor = tensor_single.repeat(32, 1, 1, 1).to(device).requires_grad_(True)
            activations.clear(); model.zero_grad(set_to_none=True)
            logits = model(tensor)
            if len(activations) != 1: raise RuntimeError(f"Número inesperado de activaciones para {image_id}.")
            activation = activations[0]
            probabilities = torch.softmax(logits / config["system"]["temperature"], dim=1)[0].detach().cpu().numpy()
            saved = predictions.loc[image_id, [f"probability_calibrated_{label}" for label in CLASS_ORDER]].to_numpy(dtype=float)
            difference = float(np.max(np.abs(probabilities - saved))); maximum_probability_difference = max(maximum_probability_difference, difference)
            predicted_index = int(logits[0].argmax().item()); true_index = int(row["true_index"])
            if predicted_index != int(row["predicted_index"]) or difference > config["gradcam"]["probability_tolerance_absolute"]:
                raise RuntimeError(f"La predicción recalculada no coincide para {image_id}: diferencia {difference}.")
            is_error = predicted_index != true_index
            predicted_cam, predicted_status = gradcam_map(logits[0, predicted_index], activation, retain_graph=is_error)
            predicted_overlay = overlay(display, predicted_cam, config["gradcam"]["overlay_opacity"])
            predicted_heatmap_path = maps_dir / f"{image_id}_predicted_{CLASS_ORDER[predicted_index]}_heatmap.png"
            predicted_overlay_path = maps_dir / f"{image_id}_predicted_{CLASS_ORDER[predicted_index]}_overlay.png"
            plt.imsave(predicted_heatmap_path, predicted_cam, cmap="jet", vmin=0, vmax=1)
            plt.imsave(predicted_overlay_path, predicted_overlay)
            map_rows.append({"image_id": image_id, "target_type": "predicted", "target_class": CLASS_ORDER[predicted_index], **predicted_status, "heatmap_file": predicted_heatmap_path.relative_to(run_dir).as_posix(), "overlay_file": predicted_overlay_path.relative_to(run_dir).as_posix()})
            true_overlay = None
            if is_error:
                true_cam, true_status = gradcam_map(logits[0, true_index], activation, retain_graph=False)
                true_overlay = overlay(display, true_cam, config["gradcam"]["overlay_opacity"])
                true_heatmap_path = maps_dir / f"{image_id}_true_{CLASS_ORDER[true_index]}_heatmap.png"
                true_overlay_path = maps_dir / f"{image_id}_true_{CLASS_ORDER[true_index]}_overlay.png"
                plt.imsave(true_heatmap_path, true_cam, cmap="jet", vmin=0, vmax=1); plt.imsave(true_overlay_path, true_overlay)
                map_rows.append({"image_id": image_id, "target_type": "true_reference", "target_class": CLASS_ORDER[true_index], **true_status, "heatmap_file": true_heatmap_path.relative_to(run_dir).as_posix(), "overlay_file": true_overlay_path.relative_to(run_dir).as_posix()})
            panel_path = panels_dir / f"{image_id}_panel.png"
            save_individual_panel(display, predicted_overlay, true_overlay, row, panel_path)
            panel_overlays[image_id] = predicted_overlay
            example_rows.append({"image_id": image_id, "predicted_index_recomputed": predicted_index, "probability_max_abs_difference": difference, "prediction_matches_saved": True, "individual_panel": panel_path.relative_to(run_dir).as_posix(), "source_file_sha256": sha256(source), "copied_original_sha256": sha256(originals_dir / source.name)})
    finally:
        hook.remove()
    map_table = pd.DataFrame(map_rows); checks_table = pd.DataFrame(example_rows)
    map_table.to_csv(run_dir / "gradcam_map_checks.csv", index=False); checks_table.to_csv(run_dir / "example_prediction_checks.csv", index=False)
    for true_class in CLASS_ORDER:
        cases = selected[selected["true_class"] == true_class]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9)); axes = axes.ravel()
        for ax in axes: ax.axis("off")
        for ax, (_, row) in zip(axes, cases.iterrows()):
            ax.imshow(panel_overlays[str(row["image_id"])]); ax.axis("off")
            state = "A" if bool(row["accepted"]) else "R"
            ax.set_title(f"{row['selection_category']}\n{row['image_id']} · {row['lesion_id']}\n{row['true_class']}→{row['predicted_class']} · {row['confidence_calibrated']:.3f} · {state}", fontsize=9)
        fig.suptitle(f"Grad-CAM sobre clase predicha · clase real {true_class}\nSelección determinista; A=aceptada, R=rechazada", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(class_panels_dir / f"panel_{true_class}.png", dpi=170); plt.close(fig)
    digest_after = state_digest(model); checkpoint_after = sha256(checkpoint)
    attribution = config["dataset_attribution"]
    attribution_lines = [
        "# Atribución de los ejemplos HAM10000", "",
        f"Conjunto: {attribution['title']}.", f"Autores: {attribution['authors']}.",
        f"Fuente: https://doi.org/{attribution['doi']}.", f"Versión: {attribution['version']}.",
        f"Licencia: {attribution['license']} ({attribution['license_url']}).", "",
        "Este directorio contiene únicamente copias de las imágenes seleccionadas mediante la regla reproducible del protocolo, necesarias para revisar los paneles Grad-CAM. No se redistribuye el resto de HAM10000.", "",
        "## Imágenes incluidas", "",
    ] + [f"- `{row.image_id}.jpg` · lesión `{row.lesion_id}` · clase real `{row.true_class}` · categoría `{row.selection_category}`." for row in selected.itertuples()]
    (run_dir / "ATTRIBUTION_LICENSE_SELECTED_IMAGES.md").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")
    availability = pd.read_csv(run_dir / "selected_examples_availability.csv")
    summary = pd.read_csv(run_dir / "subgroup_summary.csv")
    scarce = summary[summary["scarce_evidence_less_than_20_lesions"]]
    missing = availability[availability["missing_to_maximum"] > 0]
    report = [
        "# Análisis exploratorio de subgrupos y ejemplos Grad-CAM", "",
        "Este análisis reutiliza las predicciones guardadas de la evaluación final. No repite la inferencia completa y no modifica modelo, temperatura, umbral o particiones.", "",
        "## Integridad de datos", "",
        f"Las {len(pd.read_csv(run_dir / 'test_predictions_with_original_metadata.csv'))} predicciones se unieron uno a uno con los metadatos originales por `image_id`, sin pérdidas, duplicaciones ni discrepancias de etiqueta o lesión. Los valores ausentes o `unknown` se conservaron como categorías separadas y no se imputaron.", "",
        "## Subgrupos", "",
        "Se analizaron las bandas de edad menores de 40, de 40 a 59 y de 60 o más, además de desconocidos; las categorías disponibles de sexo y localización. F1 macro y exactitud equilibrada solo se calcularon cuando estaban representadas las siete clases reales. Las métricas no estimables permanecen vacías y explicadas en las tablas.", "",
        f"Se marcaron {len(scarce)} grupos con menos de 20 lesiones como evidencia escasa: " + (", ".join(f"{row.variable}={row.group} ({row.lesions} lesiones)" for row in scarce.itertuples()) if len(scarce) else "ninguno") + ".", "",
        "Las tablas de composición muestran que los grupos no tienen la misma mezcla diagnóstica. Las tablas por clase permiten examinar diferencias de recall dentro de una misma referencia sin confundirlas con la exactitud global. Los resultados son exploratorios: no atribuyen causalidad ni demuestran equidad.", "",
        "No se estimaron tono de piel, fototipo, etnia, condición socioeconómica ni variables no anotadas.", "",
        "## Selección visual reproducible", "",
        "Para resolver el solapamiento no concretado por el protocolo, los casos rechazados forman una categoría exclusiva y prioritaria; aciertos y errores contienen solo predicciones aceptadas. Dentro de cada clase y categoría se recorrió `image_id` en orden ascendente, priorizando lesiones no usadas en el panel de esa clase y rellenando de forma determinista si era necesario. Ninguna imagen se repitió y no se consultó el aspecto de los mapas para seleccionar casos.", "",
        f"Se seleccionaron {len(selected)} imágenes. " + ("Combinaciones con menos de dos casos: " + "; ".join(f"{row.true_class}/{row.selection_category}: {row.selected}" for row in missing.itertuples()) + "." if len(missing) else "Todas las combinaciones aportaron dos casos."), "",
        "## Grad-CAM", "",
        f"Se utilizó la capa `{config['gradcam']['target_layer']}`, la última convolución del bloque final de EfficientNetB0. Cada mapa combina por promedio global los gradientes del logit anterior a softmax, aplica suma ponderada y ReLU, y normaliza por mínimo y máximo de forma independiente. La superposición usa opacidad {config['gradcam']['overlay_opacity']}. En errores se generó además el mapa del logit de la clase real.", "",
        f"Se generaron {len(map_table)} mapas. Mapas nulos: {int(map_table['null_map'].sum())}; mapas no finitos: {int((~map_table['finite']).sum())}. En la generación definitiva se mantuvo la forma de lote 32 repitiendo cada ejemplo seleccionado; durante esa generación no se cargaron otras imágenes de prueba. Esta afirmación no abarca el diagnóstico previo que reconstruyó un lote original. La diferencia absoluta máxima entre las probabilidades recalculadas de los ejemplos y las guardadas fue {maximum_probability_difference:.3g}, dentro de la tolerancia de {config['gradcam']['probability_tolerance_absolute']}.", "",
        "Los mapas son visualizaciones de sensibilidad del modelo, no demuestran razonamiento clínico ni causalidad. La revisión de pelo, reglas, marcas y bordes queda expresamente pendiente en `visual_observation_checklist.csv`; no se generaron observaciones automáticas.", "",
        "## Alcance", "",
        "Este trabajo no desarrolla todavía el demostrador, no modifica el Word y no cambia ninguna decisión del sistema después de prueba.", "",
    ]
    (run_dir / "INFORME_SUBGRUPOS_GRADCAM.md").write_text("\n".join(report), encoding="utf-8")
    technical = json.loads((run_dir / "technical_checks_prepare.json").read_text())
    technical.update({
        "gradcam_generated": True, "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after == config["system"]["checkpoint_sha256"],
        "model_state_digest_unchanged": digest_before == digest_after, "model_eval_mode": not model.training,
        "all_model_parameters_require_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
        "input_gradients_enabled_only_for_selected_examples": True, "target_layer": config["gradcam"]["target_layer"],
        "gradcam_forward_batch_shape": 32, "batch_shape_reproduction": "selected image repeated 32 times; no non-selected test image loaded",
        "target_layer_type": type(target_layer).__name__, "selected_examples_loaded": len(selected),
        "full_test_inference_repeated": False, "maximum_probability_abs_difference": maximum_probability_difference,
        "prediction_tolerance": config["gradcam"]["probability_tolerance_absolute"],
        "all_example_predictions_match": bool(checks_table["prediction_matches_saved"].all()),
        "gradcam_maps": len(map_table), "null_maps": int(map_table["null_map"].sum()), "non_finite_maps": int((~map_table["finite"]).sum()),
        "model_modified": False, "temperature_modified": False, "threshold_modified": False, "partitions_modified": False,
    })
    json_dump(run_dir / "technical_checks.json", technical)
    json_dump(run_dir / "run_summary.json", {
        "status": "completed", "completed_at_utc": utc_now(), "output_directory": str(run_dir.relative_to(ROOT)),
        "selected_images": len(selected), "gradcam_maps": len(map_table), "null_maps": int(map_table["null_map"].sum()),
        "subgroup_count": len(summary), "scarce_groups": len(scarce), "full_test_inference_repeated": False,
        "checkpoint_sha256": checkpoint_after, "test_predictions_sha256": hashes["test_predictions"],
        "model_temperature_threshold_partitions_unchanged": True,
    })
    json_dump(run_dir / "environment.json", {**environment_record(), "platform": platform.platform(), "torchvision": torchvision.__version__, "device": str(device), "gpu_name": torch.cuda.get_device_name(device)})
    print(json.dumps({"RUN_DIR": str(run_dir), "selected_images": len(selected), "gradcam_maps": len(map_table), "null_maps": int(map_table["null_map"].sum()), "max_probability_difference": maximum_probability_difference}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--gradcam", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prepare:
        if args.run_dir is not None: raise SystemExit("--run-dir no se usa con --prepare")
        prepare()
    else:
        if args.run_dir is None: raise SystemExit("--run-dir es obligatorio con --gradcam")
        generate_gradcam(args.run_dir.resolve())
