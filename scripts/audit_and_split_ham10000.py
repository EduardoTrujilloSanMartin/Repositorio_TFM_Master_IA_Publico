#!/usr/bin/env python3
"""Audita HAM10000 y genera particiones reproducibles sin entrenar modelos."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SPLIT_ORDER = ["train", "selection", "calibration", "test"]
TARGETS = {"train": 0.60, "selection": 0.15, "calibration": 0.10, "test": 0.15}
SEED = 2026
ARCHIVE_MD5 = {
    "HAM10000_images_part_1.zip": "4639bfa73ab251610530a97c898e6e46",
    "HAM10000_images_part_2.zip": "da43d6cc50f6613013be07e8986b384b",
    "HAM10000_metadata.csv": "70bb67f71bd23dde0bd2eda4bdcc98e5",
}


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        smaller, larger = sorted((a, b))
        self.parent[larger] = smaller


def md5(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        root = destination.resolve()
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Ruta insegura en {archive.name}: {member.filename}")
        zipped.extractall(destination)


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_id": path.stem,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "readable": False,
        "error": "",
    }
    try:
        raw = path.read_bytes()
        result["file_sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as check:
            check.verify()
        with Image.open(io.BytesIO(raw)) as image:
            result["width"], result["height"] = image.size
            result["original_mode"] = image.mode
            rgb = image.convert("RGB")
            pixels = rgb.tobytes()
            prefix = struct.pack(">II", rgb.width, rgb.height) + b"RGB"
            result["pixel_sha256"] = hashlib.sha256(prefix + pixels).hexdigest()
            result["readable"] = True
    except Exception as error:  # se registra sin abortar la auditoría completa
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def duplicate_table(inventory: pd.DataFrame, column: str, kind: str) -> pd.DataFrame:
    valid = inventory[inventory["readable"] & inventory[column].notna()]
    groups = valid.groupby(column, sort=True)["image_id"].agg(list)
    rows = []
    for hash_value, image_ids in groups.items():
        if len(image_ids) > 1:
            rows.append(
                {
                    "duplicate_type": kind,
                    "hash": hash_value,
                    "count": len(image_ids),
                    "image_ids": "|".join(sorted(image_ids)),
                }
            )
    return pd.DataFrame(rows, columns=["duplicate_type", "hash", "count", "image_ids"])


def save_table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Sin incidencias."

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def split_groups(groups: pd.DataFrame) -> dict[str, set[str]]:
    def split(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        first, second = train_test_split(
            frame,
            test_size=test_size,
            random_state=SEED,
            shuffle=True,
            stratify=frame["dx"],
        )
        return first.copy(), second.copy()

    remaining, test = split(groups, 0.15)
    remaining, calibration = split(remaining, 0.10 / 0.85)
    train, selection = split(remaining, 0.15 / 0.75)
    return {
        "train": set(train["consolidated_group_id"]),
        "selection": set(selection["consolidated_group_id"]),
        "calibration": set(calibration["consolidated_group_id"]),
        "test": set(test["consolidated_group_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/ham10000_v1"))
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/manifests/ham10000_v1_seed2026"),
    )
    parser.add_argument(
        "--report-root", type=Path, default=Path("reports/ham10000_audit")
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    archives = raw_root / "archives"
    images_root = raw_root / "images"
    manifest_root = args.manifest_root.resolve()
    report_root = args.report_root.resolve()
    tables = report_root / "tables"
    figures = report_root / "figures"
    for directory in (manifest_root, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    archive_checks = []
    for filename, expected in ARCHIVE_MD5.items():
        path = archives / filename
        if not path.exists():
            raise FileNotFoundError(path)
        observed = md5(path)
        archive_checks.append(
            {"filename": filename, "expected_md5": expected, "observed_md5": observed, "ok": observed == expected}
        )
        if observed != expected:
            raise RuntimeError(f"Integridad incorrecta: {filename}")

    expected_images = 10_015
    existing_images = list(images_root.rglob("*.jpg")) if images_root.exists() else []
    if len(existing_images) != expected_images:
        if images_root.exists() and existing_images:
            raise RuntimeError(
                f"Extracción parcial detectada ({len(existing_images)} imágenes). "
                "Revise o elimine solo la carpeta de extracción antes de reintentar."
            )
        for archive_name in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
            safe_extract(archives / archive_name, images_root)

    image_paths = sorted(images_root.rglob("*.jpg"))
    if len(image_paths) != expected_images:
        raise RuntimeError(
            f"Se esperaban {expected_images} imágenes y se encontraron {len(image_paths)}."
        )
    for image_path in image_paths:
        image_path.chmod(0o444)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inventory = pd.DataFrame(pool.map(inspect_image, image_paths))
    inventory = inventory.sort_values("image_id").reset_index(drop=True)
    inventory.to_csv(tables / "image_inventory.csv", index=False)

    duplicate_files = duplicate_table(inventory, "file_sha256", "file_bytes")
    duplicate_pixels = duplicate_table(inventory, "pixel_sha256", "decoded_pixels")
    duplicates = pd.concat([duplicate_files, duplicate_pixels], ignore_index=True)
    duplicates.to_csv(tables / "exact_duplicate_groups.csv", index=False)

    metadata_path = archives / "HAM10000_metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"lesion_id": "string", "image_id": "string", "dx": "string"})
    required_columns = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    if not required_columns.issubset(metadata.columns):
        raise RuntimeError(f"Columnas ausentes: {sorted(required_columns - set(metadata.columns))}")

    metadata["image_id"] = metadata["image_id"].astype("string")
    metadata["lesion_id"] = metadata["lesion_id"].astype("string")
    metadata["dx"] = metadata["dx"].astype("string")
    metadata_duplicate_ids = metadata[metadata["image_id"].duplicated(keep=False)].copy()
    metadata_duplicate_ids.to_csv(tables / "duplicate_metadata_image_ids.csv", index=False)

    image_ids = set(inventory["image_id"])
    metadata_ids = set(metadata["image_id"].dropna())
    missing_files = sorted(metadata_ids - image_ids)
    extra_files = sorted(image_ids - metadata_ids)
    pd.DataFrame({"image_id": missing_files}).to_csv(tables / "metadata_without_image.csv", index=False)
    pd.DataFrame({"image_id": extra_files}).to_csv(tables / "images_without_metadata.csv", index=False)

    merged = metadata.merge(
        inventory,
        on="image_id",
        how="left",
        validate="one_to_one" if metadata_duplicate_ids.empty else "many_to_one",
    )
    merged["age_original"] = merged["age"]
    merged["age_analysis"] = pd.to_numeric(merged["age"], errors="coerce")
    age_out_of_range = merged["age_analysis"].notna() & ~merged["age_analysis"].between(0, 120)
    merged.loc[age_out_of_range, "age_analysis"] = np.nan

    missingness_rows = []
    for column in ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]:
        missing = int(merged[column].isna().sum() + (merged[column].astype("string").str.strip() == "").sum())
        explicit_unknown = int(merged[column].astype("string").str.lower().isin(["unknown", "unk"]).sum())
        missingness_rows.append(
            {
                "field": column,
                "missing_count": missing,
                "missing_percent": 100 * missing / len(merged),
                "explicit_unknown_count": explicit_unknown,
            }
        )
    missingness = pd.DataFrame(missingness_rows)
    missingness.to_csv(tables / "missingness.csv", index=False)

    contradictory_original = (
        merged.dropna(subset=["lesion_id", "dx"])
        .groupby("lesion_id")["dx"]
        .nunique()
    )
    contradictory_original_ids = set(contradictory_original[contradictory_original > 1].index.astype(str))

    valid_lesions = sorted(merged["lesion_id"].dropna().astype(str).unique())
    union_find = UnionFind(valid_lesions)
    lookup_lesion = merged.set_index("image_id")["lesion_id"].astype("string").to_dict()
    for table in (duplicate_files, duplicate_pixels):
        for image_list in table.get("image_ids", pd.Series(dtype=str)):
            lesions = sorted(
                {
                    str(lookup_lesion[image_id])
                    for image_id in image_list.split("|")
                    if image_id in lookup_lesion and pd.notna(lookup_lesion[image_id])
                }
            )
            for lesion in lesions[1:]:
                union_find.union(lesions[0], lesion)
    merged["consolidated_group_id"] = merged["lesion_id"].map(
        lambda value: union_find.find(str(value)) if pd.notna(value) else pd.NA
    )

    contradictory_consolidated = (
        merged.dropna(subset=["consolidated_group_id", "dx"])
        .groupby("consolidated_group_id")["dx"]
        .nunique()
    )
    contradictory_consolidated_ids = set(
        contradictory_consolidated[contradictory_consolidated > 1].index.astype(str)
    )

    reasons: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_metadata_set = set(metadata_duplicate_ids["image_id"].astype(str))
    for row in merged.itertuples(index=False):
        image_id = str(row.image_id)
        if image_id in duplicate_metadata_set:
            reasons[image_id].add("duplicate_image_id_in_metadata")
        if pd.isna(row.lesion_id) or not str(row.lesion_id).strip():
            reasons[image_id].add("missing_lesion_id")
        if pd.isna(row.dx) or str(row.dx) not in CLASSES:
            reasons[image_id].add("missing_or_invalid_dx")
        if pd.isna(row.path):
            reasons[image_id].add("image_file_missing")
        elif not bool(row.readable):
            reasons[image_id].add("image_unreadable")
        if pd.notna(row.lesion_id) and str(row.lesion_id) in contradictory_original_ids:
            reasons[image_id].add("contradictory_labels_within_lesion")
        if pd.notna(row.consolidated_group_id) and str(row.consolidated_group_id) in contradictory_consolidated_ids:
            reasons[image_id].add("contradictory_labels_after_duplicate_merge")

    exclusion_rows = [
        {"image_id": image_id, "reasons": "|".join(sorted(values))}
        for image_id, values in sorted(reasons.items())
        if values
    ]
    exclusions = pd.DataFrame(exclusion_rows, columns=["image_id", "reasons"])
    exclusions.to_csv(tables / "exclusions.csv", index=False)
    excluded_ids = set(exclusions["image_id"]) if not exclusions.empty else set()
    eligible = merged[~merged["image_id"].astype(str).isin(excluded_ids)].copy()

    groups = (
        eligible.groupby("consolidated_group_id", as_index=False)
        .agg(dx=("dx", "first"), image_count=("image_id", "size"), lesion_count=("lesion_id", "nunique"))
        .sort_values("consolidated_group_id")
    )
    if not (groups.groupby("consolidated_group_id")["dx"].nunique() == 1).all():
        raise RuntimeError("Persisten etiquetas contradictorias antes de particionar.")
    split_sets = split_groups(groups)
    group_to_split = {group: split_name for split_name, values in split_sets.items() for group in values}
    eligible["split"] = eligible["consolidated_group_id"].map(group_to_split)
    if eligible["split"].isna().any():
        raise RuntimeError("Hay imágenes elegibles sin partición.")

    relative_base = raw_root.parent.parent
    eligible["relative_path"] = eligible["path"].map(
        lambda value: str(Path(value).resolve().relative_to(relative_base))
    )
    manifest_columns = [
        "image_id", "lesion_id", "consolidated_group_id", "dx", "dx_type",
        "age_original", "age_analysis", "sex", "localization", "relative_path",
        "file_sha256", "pixel_sha256", "width", "height", "original_mode", "split",
    ]
    manifest = eligible[manifest_columns].sort_values(["split", "dx", "consolidated_group_id", "image_id"])
    manifest.to_csv(manifest_root / "all_splits.csv", index=False)
    for split_name in SPLIT_ORDER:
        split_frame = manifest[manifest["split"] == split_name].sort_values("image_id")
        split_frame.to_csv(manifest_root / f"{split_name}.csv", index=False)
        (manifest_root / f"{split_name}_image_ids.txt").write_text(
            "\n".join(split_frame["image_id"]) + "\n", encoding="utf-8"
        )

    class_distribution = (
        merged["dx"].value_counts().reindex(CLASSES, fill_value=0).rename_axis("dx").reset_index(name="images")
    )
    class_distribution["percent"] = 100 * class_distribution["images"] / len(merged)
    class_distribution.to_csv(tables / "class_distribution.csv", index=False)

    split_images = pd.crosstab(manifest["split"], manifest["dx"]).reindex(index=SPLIT_ORDER, columns=CLASSES, fill_value=0)
    split_images["total"] = split_images.sum(axis=1)
    split_images["target_percent"] = [100 * TARGETS[name] for name in SPLIT_ORDER]
    split_images["observed_percent"] = 100 * split_images["total"] / len(manifest)
    split_images.reset_index().to_csv(tables / "split_distribution_images.csv", index=False)

    group_frame = groups.assign(split=groups["consolidated_group_id"].map(group_to_split))
    split_groups_table = pd.crosstab(group_frame["split"], group_frame["dx"]).reindex(index=SPLIT_ORDER, columns=CLASSES, fill_value=0)
    split_groups_table["total"] = split_groups_table.sum(axis=1)
    split_groups_table["target_percent"] = [100 * TARGETS[name] for name in SPLIT_ORDER]
    split_groups_table["observed_percent"] = 100 * split_groups_table["total"] / len(group_frame)
    split_groups_table.reset_index().to_csv(tables / "split_distribution_groups.csv", index=False)

    group_overlap: dict[str, list[str]] = {}
    hash_overlap: dict[str, dict[str, list[str]]] = {}
    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1 :]:
            key = f"{left}__{right}"
            group_overlap[key] = sorted(split_sets[left] & split_sets[right])
            hash_overlap[key] = {}
            for column in ("file_sha256", "pixel_sha256"):
                a = set(manifest.loc[manifest["split"] == left, column].dropna())
                b = set(manifest.loc[manifest["split"] == right, column].dropna())
                hash_overlap[key][column] = sorted(a & b)
    all_classes_present = {
        split_name: sorted(set(manifest.loc[manifest["split"] == split_name, "dx"])) == CLASSES
        for split_name in SPLIT_ORDER
    }
    checks = {
        "seed": SEED,
        "targets": TARGETS,
        "group_intersections": group_overlap,
        "exact_hash_intersections": hash_overlap,
        "all_group_intersections_empty": all(not values for values in group_overlap.values()),
        "all_hash_intersections_empty": all(
            not values for pair in hash_overlap.values() for values in pair.values()
        ),
        "each_image_once": bool(manifest["image_id"].is_unique),
        "all_classes_present": all_classes_present,
        "test_used_for_model_or_threshold_selection": False,
        "patient_independence_claimed": False,
    }
    (manifest_root / "integrity_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=class_distribution, x="dx", y="images", hue="dx", legend=False, ax=axis)
    axis.set(title="Distribución original de HAM10000", xlabel="Clase", ylabel="Imágenes")
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=180)
    plt.close(fig)

    missingness_plot = missingness.melt(
        id_vars="field",
        value_vars=["missing_count", "explicit_unknown_count"],
        var_name="tipo",
        value_name="count",
    )
    missingness_plot["tipo"] = missingness_plot["tipo"].map(
        {"missing_count": "Ausente", "explicit_unknown_count": "unknown explícito"}
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=missingness_plot, x="field", y="count", hue="tipo", ax=axis)
    axis.set(
        title="Ausencias y valores unknown en los metadatos",
        xlabel="Campo",
        ylabel="Recuento",
    )
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "metadata_missingness.png", dpi=180)
    plt.close(fig)

    heatmap_data = split_images[CLASSES]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(title="Imágenes por clase y partición", xlabel="Clase", ylabel="Partición")
    fig.tight_layout()
    fig.savefig(figures / "split_class_counts.png", dpi=180)
    plt.close(fig)

    provenance_record = json.loads((raw_root / "provenance" / "download_record.json").read_text(encoding="utf-8"))
    incident_summary = pd.DataFrame(
        [
            ("Filas de metadatos", len(metadata)),
            ("Archivos JPEG", len(inventory)),
            ("Archivos ilegibles", int((~inventory["readable"]).sum())),
            ("Metadatos sin imagen", len(missing_files)),
            ("Imágenes sin metadatos", len(extra_files)),
            ("image_id repetidos en metadatos", len(metadata_duplicate_ids)),
            ("Grupos de bytes duplicados", len(duplicate_files)),
            ("Grupos de píxeles duplicados", len(duplicate_pixels)),
            ("Lesiones con etiquetas contradictorias", len(contradictory_original_ids)),
            ("Edades fuera de 0 a 120", int(age_out_of_range.sum())),
            ("Imágenes excluidas", len(exclusions)),
            ("Imágenes elegibles", len(manifest)),
            ("Lesiones originales elegibles", int(eligible["lesion_id"].nunique())),
            ("Grupos consolidados elegibles", len(groups)),
        ],
        columns=["Comprobación", "Resultado"],
    )
    incident_summary.to_csv(tables / "incident_summary.csv", index=False)

    summary = {
        "source": provenance_record,
        "archive_integrity": archive_checks,
        "metadata_rows": len(metadata),
        "jpeg_files": len(inventory),
        "eligible_images": len(manifest),
        "excluded_images": len(exclusions),
        "eligible_original_lesions": int(eligible["lesion_id"].nunique()),
        "eligible_consolidated_groups": len(groups),
        "exact_file_duplicate_groups": len(duplicate_files),
        "exact_pixel_duplicate_groups": len(duplicate_pixels),
        "contradictory_lesion_ids": sorted(contradictory_original_ids),
        "checks": checks,
    }
    (report_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = f"""# Auditoría y particiones de HAM10000

Fecha de descarga: {provenance_record['download_date_local']}

## Procedencia y alcance

Se descargó la versión 1.0 original de HAM10000 desde Harvard Dataverse, DOI `{provenance_record['persistent_id']}`. La colección está sujeta a la licencia Creative Commons Attribution-NonCommercial 4.0 International. Se conservaron los dos ZIP, el CSV original y el TAB derivado por Dataverse en modo de solo lectura. No se incorporaron las segmentaciones ni los conjuntos de evaluación añadidos en versiones posteriores del registro.

El conjunto no contiene un identificador fiable de paciente. Las particiones están agrupadas por `lesion_id` y no se afirma independencia entre pacientes.

## Integridad y correspondencia

{save_table_markdown(incident_summary)}

Las huellas MD5 de los dos ZIP y del CSV original coinciden con Harvard Dataverse. El TAB servido por defecto es una representación derivada por Dataverse y se conserva separado. Los archivos originales no se han transformado; la extracción conserva los bytes JPEG contenidos en los ZIP.

## Distribución de clases

{save_table_markdown(class_distribution.round({'percent': 3}))}

![Distribución de clases](figures/class_distribution.png)

## Valores ausentes

{save_table_markdown(missingness.round({'missing_percent': 3}))}

Los valores ausentes se mantienen como tales. Los valores explícitos `unknown` se contabilizan por separado. No se imputaron edad, sexo ni localización; las edades fuera de 0 a 120, si existen, solo se anulan en `age_analysis` y se preservan en `age_original`.

![Ausencias y valores unknown](figures/metadata_missingness.png)

## Exclusiones e incidencias

Las exclusiones detalladas están en `tables/exclusions.csv`. Una imagen solo se excluye por identificador o etiqueta imprescindible ausente o inválida, archivo inexistente o ilegible, duplicidad de `image_id` en metadatos, o contradicción de etiqueta dentro de su grupo. Los metadatos clínicos ausentes no causan exclusión.

Los duplicados exactos se comprobaron tanto sobre los bytes del archivo como sobre los píxeles RGB decodificados. Cuando un duplicado exacto enlaza lesiones diferentes, sus grupos se consolidan antes de particionar. No se ha realizado una búsqueda de similitud visual o casi duplicados en esta ejecución; esa revisión no altera la comprobación exigida de duplicados exactos.

## Particiones congeladas

Semilla: `{SEED}`. Proporciones objetivo por grupos de lesión: entrenamiento 60 %, selección 15 %, calibración 10 % y prueba 15 %.

### Por imágenes

{save_table_markdown(split_images.reset_index().round({'target_percent': 3, 'observed_percent': 3}))}

### Por grupos consolidados

{save_table_markdown(split_groups_table.reset_index().round({'target_percent': 3, 'observed_percent': 3}))}

![Distribución por partición y clase](figures/split_class_counts.png)

Las intersecciones de grupos y de huellas exactas entre particiones están vacías: `{checks['all_group_intersections_empty'] and checks['all_hash_intersections_empty']}`. Cada imagen elegible aparece una vez: `{checks['each_image_once']}`. Las siete clases aparecen en todas las particiones: `{all(checks['all_classes_present'].values())}`.

El conjunto de prueba se ha limitado a la asignación reproducible y a comprobaciones de integridad y recuento. No se ha utilizado para seleccionar modelos, umbrales, transformaciones ni otras decisiones.

## Archivos reproducibles

- `scripts/download_ham10000.py`: descarga y verificación de originales.
- `scripts/audit_and_split_ham10000.py`: auditoría, exclusiones, particiones y figuras.
- `data/manifests/ham10000_v1_seed2026/`: manifiesto completo, listados por partición y comprobaciones.
- `reports/ham10000_audit/tables/`: inventario y tablas de auditoría.
"""
    (report_root / "INFORME_AUDITORIA_HAM10000.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
