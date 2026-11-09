#!/usr/bin/env python3
"""Comprueba la ruta compartida del demostrador contra 39 predicciones guardadas."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from inference_core import (
    CLASS_ORDER,
    DEFAULT_CONFIG,
    ROOT,
    InferenceEngine,
    InvalidImageError,
    decision_from_confidence,
    sha256,
)


OUTPUT_ROOT = ROOT / "experiments/demonstrator"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = OUTPUT_ROOT / f"validation_primary_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def invalid_file_checks(engine: InferenceEngine) -> list[dict[str, object]]:
    values: list[tuple[str, bytes]] = [("empty", b""), ("text", b"not an image")]
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (100, 50, 25)).save(buffer, format="BMP")
    values.append(("unsupported_bmp", buffer.getvalue()))
    source = ROOT / "experiments/subgroups_gradcam/primary_test_20261021_162747/selected_originals/ISIC_0024923.jpg"
    values.append(("truncated_jpeg", source.read_bytes()[:128]))
    rows = []
    for name, payload in values:
        rejected = False
        message = ""
        try:
            engine.decode_and_preprocess(payload)
        except InvalidImageError as error:
            rejected = True
            message = str(error)
        rows.append({"case": name, "rejected": rejected, "message": message})
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("La validación formal del demostrador requiere CUDA.")
    output = unique_output()
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    checkpoint = ROOT / config["system"]["checkpoint"]
    checkpoint_before = sha256(checkpoint)
    engine = InferenceEngine()
    selected = pd.read_csv(ROOT / config["validation"]["selected_examples"])
    saved = pd.read_csv(ROOT / config["validation"]["saved_predictions"]).set_index("image_id")
    if len(selected) != config["validation"]["selected_count"] or selected["image_id"].duplicated().any():
        raise RuntimeError("La selección de 39 ejemplos no coincide con la configuración.")
    rows: list[dict[str, object]] = []
    maximum_difference = 0.0
    null_maps = 0
    non_finite_maps = 0
    for _, selected_row in selected.sort_values("image_id").iterrows():
        image_id = str(selected_row["image_id"])
        source = ROOT / "data" / str(selected_row["relative_path"])
        result = engine.predict(source, with_gradcam=True)
        expected = saved.loc[
            image_id, [f"probability_calibrated_{label}" for label in CLASS_ORDER]
        ].to_numpy(dtype=float)
        difference = float(np.max(np.abs(result.probabilities.astype(float) - expected)))
        maximum_difference = max(maximum_difference, difference)
        expected_class = str(saved.loc[image_id, "predicted_label"])
        expected_accepted = bool(saved.loc[image_id, "accepted"])
        class_matches = result.predicted_class == expected_class
        decision_matches = result.accepted == expected_accepted
        probability_sum = float(result.probabilities.sum())
        if difference > config["inference"]["probability_tolerance_absolute"]:
            raise RuntimeError(f"{image_id}: diferencia {difference} fuera de tolerancia.")
        if not class_matches or not decision_matches:
            raise RuntimeError(f"{image_id}: clase o abstención no coinciden.")
        if not np.isclose(probability_sum, 1.0, atol=1e-6):
            raise RuntimeError(f"{image_id}: las probabilidades no suman uno.")
        null_maps += int(bool(result.null_map))
        non_finite_maps += int(not bool(result.map_finite))
        rows.append(
            {
                "image_id": image_id,
                "source_sha256": sha256(source),
                "max_abs_probability_difference": difference,
                "probability_sum": probability_sum,
                "predicted_class_shared_route": result.predicted_class,
                "predicted_class_saved": expected_class,
                "class_matches": class_matches,
                "confidence_shared_route": result.confidence,
                "accepted_shared_route": result.accepted,
                "accepted_saved": expected_accepted,
                "decision_matches": decision_matches,
                "gradcam_finite": result.map_finite,
                "gradcam_null": result.null_map,
            }
        )
    invalid = invalid_file_checks(engine)
    if not all(bool(item["rejected"]) for item in invalid):
        raise RuntimeError("Algún archivo inválido no fue rechazado.")
    threshold = float(config["system"]["threshold"])
    boundary = {
        "threshold": threshold,
        "equal_is_accepted": decision_from_confidence(threshold, threshold),
        "next_float_below_is_rejected": not decision_from_confidence(
            float(np.nextafter(threshold, -np.inf)), threshold
        ),
        "next_float_above_is_accepted": decision_from_confidence(
            float(np.nextafter(threshold, np.inf)), threshold
        ),
    }
    if not all(boundary[key] for key in boundary if key != "threshold"):
        raise RuntimeError("La comprobación de igualdad con el umbral falló.")
    checkpoint_after = sha256(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("El checkpoint cambió durante la comprobación.")
    pd.DataFrame(rows).to_csv(output / "selected_examples_route_checks.csv", index=False)
    pd.DataFrame(invalid).to_csv(output / "invalid_file_checks.csv", index=False)
    (output / "threshold_boundary_check.json").write_text(
        json.dumps(boundary, indent=2) + "\n", encoding="utf-8"
    )
    technical = {
        "completed_at_utc": utc_now(),
        "status": "passed",
        "cuda_device": torch.cuda.get_device_name(0),
        "selected_examples": len(rows),
        "forward_batch_shape": engine.default_batch_shape,
        "batch_shape_method": config["inference"]["batch_shape_method"],
        "maximum_absolute_probability_difference": maximum_difference,
        "probability_tolerance_absolute": config["inference"]["probability_tolerance_absolute"],
        "all_classes_match": all(bool(row["class_matches"]) for row in rows),
        "all_abstention_decisions_match": all(bool(row["decision_matches"]) for row in rows),
        "all_probability_sums_valid": all(abs(float(row["probability_sum"]) - 1.0) <= 1e-6 for row in rows),
        "gradcam_maps_checked": len(rows),
        "gradcam_null_maps": null_maps,
        "gradcam_non_finite_maps": non_finite_maps,
        "invalid_files_rejected": len(invalid),
        "threshold_boundary": boundary,
        "class_order": engine.class_order,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "config_sha256": sha256(DEFAULT_CONFIG),
        "shared_route_script_sha256": sha256(ROOT / "scripts/inference_core.py"),
        "test_script_sha256": sha256(Path(__file__)),
        "full_test_inference_repeated": False,
        "only_selected_test_images_loaded": True,
        "training_or_recalibration_performed": False,
    }
    (output / "technical_checks.json").write_text(
        json.dumps(technical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Comprobación de la ruta compartida del demostrador

La ruta compartida se comprobó sobre los 39 ejemplos seleccionados, sin repetir la inferencia completa de prueba. Cada imagen se preprocesó con la configuración congelada y se repitió hasta formar un lote de 32; se utilizó la fila cero. Esta decisión reproduce la forma de lote CUDA del evaluador final, pero multiplica por 32 el trabajo de la red para una sola solicitud y su latencia se mide por separado.

La diferencia absoluta máxima frente a las probabilidades calibradas guardadas fue `{maximum_difference:.12g}`, con tolerancia absoluta `0.00001`. Coincidieron las 39 clases y las 39 decisiones de aceptación o abstención. Las probabilidades conservaron el orden `{', '.join(CLASS_ORDER)}` y sumaron uno dentro de tolerancia.

Se generó Grad-CAM para los 39 ejemplos mediante la misma ruta. Mapas nulos: {null_maps}; mapas no finitos: {non_finite_maps}. Se rechazaron correctamente cuatro entradas inválidas: archivo vacío, texto, BMP no admitido y JPEG truncado. La igualdad exacta con el umbral se acepta, conforme a `confianza >= umbral`.

No se entrenó, recalibró o adaptó el umbral. El checkpoint conservó SHA-256 `{checkpoint_after}`.
"""
    (output / "INFORME_COMPROBACION_DEMOSTRADOR.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), **technical}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
