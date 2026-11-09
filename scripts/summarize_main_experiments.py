#!/usr/bin/env python3
"""Aggregate the nine selection runs without opening calibration or test data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "main_experiments"
ARCHS = ["cnn_simple", "mobilenet_v3_small", "efficientnet_b0"]
ARCH_LABEL = {
    "cnn_simple": "CNN sencilla",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNetB0",
}
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXPECTED_HASHES = {
    "data/manifests/ham10000_v1_seed2026/train.csv": "dffa8afe7d425717b4594ef7b6ee95adf1ca191faccdc1af7635de6f6a5966d7",
    "data/manifests/ham10000_v1_seed2026/selection.csv": "5563d434fad5182f2ca46ae80ab6ecbb86161aa7532d9a7f11434f69a35c4149",
}


def one_run(arch: str, seed: int) -> Path:
    matches = sorted((ROOT / "experiments" / arch).glob(f"formal_seed{seed}_*/run_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one run for {arch} seed {seed}, found {len(matches)}")
    return matches[0].parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Manifest changed: {rel}: {actual}")

    rows, class_rows, auc_rows, registry = [], [], [], []
    for arch in ARCHS:
        for seed in (17, 29, 43):
            run = one_run(arch, seed)
            summary = json.loads((run / "run_summary.json").read_text())
            if summary["status"] != "completed" or summary["calibration_used"] or summary["test_used"]:
                raise RuntimeError(f"Invalid run state: {run}")
            if summary["manifest_sha256"] != EXPECTED_HASHES:
                raise RuntimeError(f"Manifest hash mismatch in {run}")
            metrics = summary["best_selection_metrics"]
            pred = pd.read_csv(run / "selection_predictions.csv")
            y = pred["true_index"].to_numpy()
            prob = pred[[f"probability_{c}" for c in CLASSES]].to_numpy()
            per_auc = roc_auc_score(y, prob, labels=np.arange(7), multi_class="ovr", average=None)
            macro_auc = float(np.mean(per_auc))
            row = {
                "architecture": arch,
                "architecture_label": ARCH_LABEL[arch],
                "seed": seed,
                "f1_macro": metrics["f1_macro"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "roc_auc_ovr_macro": macro_auc,
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "stopped_early": summary["stopped_early"],
                "time_seconds": summary["total_training_and_epoch_validation_seconds"],
                "peak_allocated_mib": summary["process_peak_allocated_mib"],
                "peak_reserved_mib": summary["process_peak_reserved_mib"],
                "device_peak_used_mib_nvml": summary["device_peak_used_mib_nvml"],
                "parameter_count": summary["parameter_count"],
                "run_directory": str(run.relative_to(ROOT)),
            }
            rows.append(row)
            registry.append({"architecture": arch, "seed": seed, "run_directory": str(run.relative_to(ROOT))})
            for cls, auc in zip(CLASSES, per_auc):
                cm = metrics["per_class"][cls]
                class_rows.append({"architecture": arch, "seed": seed, "class": cls, **cm})
                auc_rows.append({"architecture": arch, "seed": seed, "class": cls, "roc_auc_ovr": auc})

    runs = pd.DataFrame(rows)
    runs["architecture"] = pd.Categorical(runs["architecture"], categories=ARCHS, ordered=True)
    runs = runs.sort_values(["architecture", "seed"])
    per_class = pd.DataFrame(class_rows).sort_values(["architecture", "seed", "class"])
    aucs = pd.DataFrame(auc_rows).sort_values(["architecture", "seed", "class"])
    runs.to_csv(OUT / "comparison_nine_runs.csv", index=False)
    per_class.to_csv(OUT / "per_class_nine_runs.csv", index=False)
    aucs.to_csv(OUT / "per_class_roc_auc_nine_runs.csv", index=False)
    (OUT / "run_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    agg = runs.groupby(["architecture", "architecture_label"], sort=False).agg(
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        roc_auc_ovr_macro_mean=("roc_auc_ovr_macro", "mean"),
        time_seconds_mean=("time_seconds", "mean"),
        peak_reserved_mib_max=("peak_reserved_mib", "max"),
        device_peak_used_mib_nvml_max=("device_peak_used_mib_nvml", "max"),
        parameter_count=("parameter_count", "first"),
    ).reset_index()
    agg.to_csv(OUT / "architecture_summary.csv", index=False)

    top_rounded = agg["f1_macro_mean"].round(3).max()
    candidates = agg[agg["f1_macro_mean"].round(3) == top_rounded]
    chosen = candidates.sort_values(["parameter_count", "architecture"]).iloc[0]
    chosen_arch = str(chosen["architecture"])
    arch_runs = runs[runs["architecture"] == chosen_arch].copy()
    median = float(arch_runs["f1_macro"].median())
    arch_runs["distance_to_median"] = (arch_runs["f1_macro"] - median).abs()
    representative = arch_runs.sort_values(["distance_to_median", "seed"]).iloc[0]
    decision = {
        "rule": "mayor media de F1 macro en selección; empate a 3 decimales: menos parámetros",
        "representative_rule": "F1 macro más próximo a la mediana de la arquitectura; empate: semilla menor",
        "selected_architecture": chosen_arch,
        "selected_architecture_label": ARCH_LABEL[chosen_arch],
        "selected_mean_f1_macro": float(chosen["f1_macro_mean"]),
        "selected_std_f1_macro_sample": float(chosen["f1_macro_std"]),
        "median_f1_macro": median,
        "representative_seed": int(representative["seed"]),
        "representative_f1_macro": float(representative["f1_macro"]),
        "representative_run_directory": representative["run_directory"],
        "representative_checkpoint": representative["run_directory"] + "/best_checkpoint.pt",
        "calibration_used": False,
        "test_used": False,
    }
    (OUT / "selection_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    order = [ARCH_LABEL[a] for a in ARCHS]
    sns.barplot(data=runs, x="architecture_label", y="f1_macro", order=order, errorbar="sd", capsize=.15, alpha=.55, ax=ax)
    sns.stripplot(data=runs, x="architecture_label", y="f1_macro", order=order, hue="seed", palette="deep", size=7, ax=ax)
    ax.set(xlabel="Arquitectura", ylabel="F1 macro de selección", title="Media, desviación estándar y semillas")
    ax.legend(title="Semilla", loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "comparison_f1_macro.png", dpi=180); plt.close(fig)

    recall = per_class.groupby(["architecture", "class"], sort=False)["recall"].mean().unstack()
    recall = recall.reindex(index=ARCHS, columns=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.heatmap(recall, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis", ax=ax)
    ax.set_yticklabels([ARCH_LABEL[a] for a in recall.index], rotation=0)
    ax.set(xlabel="Clase", ylabel="Arquitectura", title="Recall medio de selección por clase")
    fig.tight_layout(); fig.savefig(OUT / "mean_recall_per_class.png", dpi=180); plt.close(fig)

    def n(x): return f"{x:.6f}".replace(".", ",")
    lines = [
        "# Informe de los experimentos principales",
        "",
        "**Ámbito:** resultados de selección. No constituyen evaluación final; calibración y prueba final no se consultaron.",
        "",
        "## Nueve ejecuciones",
        "",
        "| Arquitectura | Semilla | F1 macro | Balanced acc. | AUC OVR macro | Mejor época | Épocas | Tiempo (s) | Pico reservado (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in runs.iterrows():
        lines.append(f"| {r.architecture_label} | {r.seed} | {n(r.f1_macro)} | {n(r.balanced_accuracy)} | {n(r.roc_auc_ovr_macro)} | {r.best_epoch} | {r.epochs_completed} | {r.time_seconds:.2f} | {r.peak_reserved_mib:.1f} |")
    lines += ["", "## Resumen por arquitectura", "", "La desviación estándar indicada es muestral (n = 3).", "", "| Arquitectura | F1 macro media ± DE | Balanced acc. media ± DE | Parámetros |", "| --- | ---: | ---: | ---: |"]
    for _, r in agg.iterrows():
        lines.append(f"| {r.architecture_label} | {n(r.f1_macro_mean)} ± {n(r.f1_macro_std)} | {n(r.balanced_accuracy_mean)} ± {n(r.balanced_accuracy_std)} | {int(r.parameter_count):,} |".replace(",", ".", 1) if False else f"| {r.architecture_label} | {n(r.f1_macro_mean)} ± {n(r.f1_macro_std)} | {n(r.balanced_accuracy_mean)} ± {n(r.balanced_accuracy_std)} | {int(r.parameter_count):,} |")
    lines += [
        "", "## Selección conforme al protocolo", "",
        f"Se selecciona **{decision['selected_architecture_label']}** por su F1 macro medio de selección ({n(decision['selected_mean_f1_macro'])} ± {n(decision['selected_std_f1_macro_sample'])}).",
        f"El checkpoint representativo es la semilla **{decision['representative_seed']}**, por ser el F1 más próximo a la mediana de la arquitectura ({n(decision['median_f1_macro'])}). No se eligió simplemente la mejor semilla.",
        "", "Las métricas completas por clase están en `per_class_nine_runs.csv`; las predicciones y matrices de cada ejecución permanecen en sus directorios. No se detectó ninguna incompatibilidad que exigiera modificar el protocolo.",
    ]
    (OUT / "INFORME_EXPERIMENTOS_PRINCIPALES.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
