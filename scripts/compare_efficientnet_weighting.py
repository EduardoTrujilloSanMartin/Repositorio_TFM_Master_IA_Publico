#!/usr/bin/env python3
"""Compare weighted and unweighted EfficientNetB0 runs using selection only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "efficientnet_weighting_comparison"
SEEDS = (17, 29, 43)
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MINORITY = ("akiec", "bcc", "df", "vasc")
HASHES = {
    "data/manifests/ham10000_v1_seed2026/train.csv": "dffa8afe7d425717b4594ef7b6ee95adf1ca191faccdc1af7635de6f6a5966d7",
    "data/manifests/ham10000_v1_seed2026/selection.csv": "5563d434fad5182f2ca46ae80ab6ecbb86161aa7532d9a7f11434f69a35c4149",
}


def run_dir(folder: str, seed: int) -> Path:
    matches = sorted((ROOT / "experiments" / folder).glob(f"formal_seed{seed}_*/run_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Se esperaba una ejecución para {folder}, semilla {seed}; encontradas {len(matches)}")
    return matches[0].parent


def decimal(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "weighted": ("Ponderada", "efficientnet_b0"),
        "unweighted": ("Sin ponderación", "efficientnet_b0_unweighted"),
    }
    rows: list[dict] = []
    class_rows: list[dict] = []
    fn_rows: list[dict] = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    registry: list[dict] = []

    for variant, (label, folder) in variants.items():
        for seed in SEEDS:
            run = run_dir(folder, seed)
            summary = json.loads((run / "run_summary.json").read_text())
            if summary["status"] != "completed" or summary["calibration_used"] or summary["test_used"]:
                raise RuntimeError(f"Estado o alcance inválido: {run}")
            if summary["manifest_sha256"] != HASHES:
                raise RuntimeError(f"Hashes no conformes: {run}")
            expected_weighted = variant == "weighted"
            if variant == "unweighted":
                if summary.get("loss") != "cross_entropy_unweighted" or summary.get("class_weights_applied") is not False or summary.get("oversampling") is not False:
                    raise RuntimeError(f"La variante sin ponderación no está identificada correctamente: {run}")
            metrics = summary["best_selection_metrics"]
            mel_recall = metrics["per_class"]["mel"]["recall"]
            rows.append({
                "variant": variant, "variant_label": label, "seed": seed,
                "f1_macro": metrics["f1_macro"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "melanoma_recall": mel_recall,
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "stopped_early": summary["stopped_early"],
                "time_seconds": summary["total_training_and_epoch_validation_seconds"],
                "parameter_count": summary["parameter_count"],
                "peak_allocated_mib": summary["process_peak_allocated_mib"],
                "peak_reserved_mib": summary["process_peak_reserved_mib"],
                "device_peak_used_mib_nvml": summary["device_peak_used_mib_nvml"],
                "run_directory": str(run.relative_to(ROOT)),
            })
            for cls in CLASSES:
                cm = metrics["per_class"][cls]
                class_rows.append({"variant": variant, "variant_label": label, "seed": seed, "class": cls, **cm})
            pred = pd.read_csv(run / "selection_predictions.csv")
            predictions[(variant, seed)] = pred
            mel_fn = pred[(pred.true_label == "mel") & (pred.predicted_label != "mel")]
            counts = mel_fn.predicted_label.value_counts().reindex(CLASSES, fill_value=0)
            for destination, count in counts.items():
                if destination != "mel":
                    fn_rows.append({
                        "variant": variant, "variant_label": label, "seed": seed,
                        "destination": destination, "false_negative_count": int(count),
                        "melanoma_support": int((pred.true_label == "mel").sum()),
                        "fraction_of_all_melanoma": float(count / (pred.true_label == "mel").sum()),
                    })
            registry.append({"variant": variant, "seed": seed, "run_directory": str(run.relative_to(ROOT)), "weighted": expected_weighted})

    runs = pd.DataFrame(rows)
    runs["variant"] = pd.Categorical(runs["variant"], categories=["weighted", "unweighted"], ordered=True)
    runs = runs.sort_values(["variant", "seed"])
    per_class = pd.DataFrame(class_rows).sort_values(["variant", "seed", "class"])
    fn = pd.DataFrame(fn_rows).sort_values(["variant", "seed", "destination"])
    runs.to_csv(OUT / "comparison_by_seed.csv", index=False)
    per_class.to_csv(OUT / "per_class_by_seed.csv", index=False)
    fn.to_csv(OUT / "melanoma_false_negative_destinations_by_seed.csv", index=False)
    (OUT / "run_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")

    metric_cols = ["f1_macro", "balanced_accuracy", "accuracy", "melanoma_recall", "time_seconds", "peak_reserved_mib"]
    summary = runs.groupby(["variant", "variant_label"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "comparison_summary_mean_sd.csv", index=False)

    wide = runs.pivot(index="seed", columns="variant", values=["f1_macro", "balanced_accuracy", "accuracy", "melanoma_recall"])
    delta_rows = []
    for seed in SEEDS:
        delta_rows.append({"seed": seed, **{f"delta_{metric}_unweighted_minus_weighted": float(wide.loc[seed, (metric, "unweighted")] - wide.loc[seed, (metric, "weighted")]) for metric in ["f1_macro", "balanced_accuracy", "accuracy", "melanoma_recall"]}})
    deltas = pd.DataFrame(delta_rows)
    mean_delta = {column: float(deltas[column].mean()) for column in deltas.columns if column != "seed"}
    deltas.to_csv(OUT / "paired_differences_by_seed.csv", index=False)
    (OUT / "mean_paired_differences.json").write_text(json.dumps(mean_delta, indent=2, ensure_ascii=False) + "\n")

    class_summary = per_class.groupby(["variant", "variant_label", "class"], sort=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), support=("support", "first"),
    ).reset_index()
    class_summary.to_csv(OUT / "per_class_summary_mean_sd.csv", index=False)
    cwide = class_summary.pivot(index="class", columns="variant", values=["precision_mean", "recall_mean", "f1_mean"])
    class_deltas = pd.DataFrame({
        "class": cwide.index,
        "delta_precision_unweighted_minus_weighted": cwide[("precision_mean", "unweighted")] - cwide[("precision_mean", "weighted")],
        "delta_recall_unweighted_minus_weighted": cwide[("recall_mean", "unweighted")] - cwide[("recall_mean", "weighted")],
        "delta_f1_unweighted_minus_weighted": cwide[("f1_mean", "unweighted")] - cwide[("f1_mean", "weighted")],
    }).reset_index(drop=True)
    class_deltas.to_csv(OUT / "per_class_mean_differences.csv", index=False)

    fn_summary = fn.groupby(["variant", "variant_label", "destination"], sort=False).agg(
        false_negative_count_mean=("false_negative_count", "mean"),
        false_negative_count_std=("false_negative_count", "std"),
        fraction_of_all_melanoma_mean=("fraction_of_all_melanoma", "mean"),
    ).reset_index()
    fn_summary.to_csv(OUT / "melanoma_false_negative_destinations_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    metric_long = runs.melt(id_vars=["variant_label", "seed"], value_vars=["f1_macro", "balanced_accuracy", "accuracy", "melanoma_recall"], var_name="metric", value_name="value")
    metric_names = {"f1_macro": "F1 macro", "balanced_accuracy": "Balanced accuracy", "accuracy": "Exactitud", "melanoma_recall": "Sensibilidad melanoma"}
    metric_long["metric"] = metric_long["metric"].map(metric_names)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    hue_order = ["Ponderada", "Sin ponderación"]
    sns.barplot(data=metric_long, x="metric", y="value", hue="variant_label", hue_order=hue_order, errorbar="sd", capsize=.1, ax=ax)
    ax.set(xlabel="Métrica de selección", ylabel="Media ± DE", title="EfficientNetB0: ponderada frente a sin ponderación", ylim=(0, 1))
    ax.legend(title="Variante")
    fig.tight_layout(); fig.savefig(OUT / "comparison_metrics.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.barplot(data=per_class, x="class", y="recall", hue="variant_label", hue_order=hue_order, errorbar="sd", capsize=.08, ax=ax)
    ax.set(xlabel="Clase", ylabel="Recall medio ± DE", title="Recall de selección por clase", ylim=(0, 1))
    ax.legend(title="Variante")
    fig.tight_layout(); fig.savefig(OUT / "per_class_recall_comparison.png", dpi=180); plt.close(fig)

    fn_plot = fn_summary[fn_summary.false_negative_count_mean > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=fn_plot, x="destination", y="false_negative_count_mean", hue="variant_label", hue_order=hue_order, ax=ax)
    ax.set(xlabel="Clase predicha", ylabel="FN de melanoma medios (n=3)", title="Destinos de los falsos negativos de melanoma")
    ax.legend(title="Variante")
    fig.tight_layout(); fig.savefig(OUT / "melanoma_false_negative_destinations.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, variant in zip(axes, ("weighted", "unweighted")):
        y_true = pd.concat([predictions[(variant, s)]["true_index"] for s in SEEDS], ignore_index=True)
        y_pred = pd.concat([predictions[(variant, s)]["predicted_index"] for s in SEEDS], ignore_index=True)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(7), normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=variant == "unweighted")
        ax.set(title=variants[variant][0], xlabel="Predicción", ylabel="Clase real")
    fig.suptitle("Matrices de confusión agregadas de selección (normalizadas por clase real)")
    fig.tight_layout(); fig.savefig(OUT / "aggregate_confusion_matrices.png", dpi=180); plt.close(fig)

    means = summary.set_index("variant")
    pc = class_summary.set_index(["variant", "class"])
    fn_agg = fn.groupby(["variant", "destination"])["false_negative_count"].sum()
    lines = [
        "# Comparación secundaria: EfficientNetB0 con y sin ponderación",
        "",
        "**Ámbito:** entrenamiento y selección. No se consultaron calibración ni prueba final. La selección principal permanece en EfficientNetB0 ponderada, semilla 43, época 21.",
        "",
        "La única diferencia experimental fue retirar los pesos de clase de la entropía cruzada. No se utilizó sobremuestreo; los lotes de entrenamiento se barajaron sin reemplazo.",
        "",
        "## Resultados por semilla",
        "",
        "| Variante | Semilla | F1 macro | Balanced acc. | Exactitud | Sensibilidad melanoma | Mejor época | Épocas | Tiempo (s) | Pico reservado (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in runs.iterrows():
        lines.append(f"| {r.variant_label} | {r.seed} | {decimal(r.f1_macro)} | {decimal(r.balanced_accuracy)} | {decimal(r.accuracy)} | {decimal(r.melanoma_recall)} | {r.best_epoch} | {r.epochs_completed} | {r.time_seconds:.2f} | {r.peak_reserved_mib:.1f} |")
    lines += ["", "## Media y desviación estándar muestral", "", "| Variante | F1 macro | Balanced acc. | Exactitud | Sensibilidad melanoma |", "| --- | ---: | ---: | ---: | ---: |"]
    for variant in ("weighted", "unweighted"):
        r = means.loc[variant]
        lines.append(f"| {r.variant_label} | {decimal(r.f1_macro_mean)} ± {decimal(r.f1_macro_std)} | {decimal(r.balanced_accuracy_mean)} ± {decimal(r.balanced_accuracy_std)} | {decimal(r.accuracy_mean)} ± {decimal(r.accuracy_std)} | {decimal(r.melanoma_recall_mean)} ± {decimal(r.melanoma_recall_std)} |")
    lines += [
        "", "## Diferencia media emparejada (sin ponderación − ponderada)", "",
        f"- F1 macro: **{mean_delta['delta_f1_macro_unweighted_minus_weighted']:+.6f}**.",
        f"- Balanced accuracy: **{mean_delta['delta_balanced_accuracy_unweighted_minus_weighted']:+.6f}**.",
        f"- Exactitud: **{mean_delta['delta_accuracy_unweighted_minus_weighted']:+.6f}**.",
        f"- Sensibilidad de melanoma: **{mean_delta['delta_melanoma_recall_unweighted_minus_weighted']:+.6f}**.",
        "", "## Clases minoritarias", "",
        "Las cifras siguientes son medias de selección sobre las tres semillas. No tienen interpretación clínica.", "",
        "| Clase | Recall ponderado | Recall sin ponderar | Δ recall | F1 ponderado | F1 sin ponderar | Δ F1 | Soporte |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cls in MINORITY:
        w, u = pc.loc[("weighted", cls)], pc.loc[("unweighted", cls)]
        lines.append(f"| {cls} | {decimal(w.recall_mean)} | {decimal(u.recall_mean)} | {u.recall_mean-w.recall_mean:+.6f} | {decimal(w.f1_mean)} | {decimal(u.f1_mean)} | {u.f1_mean-w.f1_mean:+.6f} | {int(w.support)} |")
    lines += ["", "## Falsos negativos de melanoma", "", "Suma de destinos en las tres semillas (cada semilla contiene 159 melanomas):", "", "| Variante | akiec | bcc | bkl | df | nv | vasc |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in ("weighted", "unweighted"):
        values = [int(fn_agg.get((variant, cls), 0)) for cls in ("akiec", "bcc", "bkl", "df", "nv", "vasc")]
        lines.append(f"| {variants[variant][0]} | " + " | ".join(map(str, values)) + " |")
    lines += [
        "", "En ambas variantes, `nv` y `bkl` concentran la mayoría de los falsos negativos de melanoma. Al retirar la ponderación aumentó la exactitud media, pero disminuyeron la balanced accuracy y la sensibilidad media de melanoma. Algunas métricas de precisión de clases minoritarias mejoraron, mientras varios recalls disminuyeron; esto refleja el compromiso estadístico esperado en un conjunto desequilibrado, sin permitir conclusiones clínicas.",
        "", "## Decisión", "",
        "Esta comparación es secundaria y **no cambia** la selección principal: EfficientNetB0 ponderada, semilla 43, época 21. No se inició calibración, abstención ni evaluación final.",
    ]
    (OUT / "INFORME_COMPARACION_PONDERACION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(OUT), "mean_differences": mean_delta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
