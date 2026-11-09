#!/usr/bin/env python3
"""Mide preprocesamiento, inferencia y Grad-CAM con el protocolo congelado."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from inference_core import (
    DEFAULT_CONFIG,
    ROOT,
    decode_image,
    load_benchmark_model,
    preprocess_image,
    sha256,
)


OUTPUT_ROOT = ROOT / "experiments/inference_latency"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_iterations(
    function: Callable[[], object], warmups: int, measurements: int, device: torch.device
) -> list[float]:
    for _ in range(warmups):
        function()
    synchronize(device)
    values = []
    for _ in range(measurements):
        synchronize(device)
        start = time.perf_counter_ns()
        function()
        synchronize(device)
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return values


def summarize(
    architecture: str,
    component: str,
    batch_shape: int,
    values: list[float],
    parameters: int | None,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "component": component,
        "batch_shape": batch_shape,
        "warmup_iterations": 20,
        "measured_iterations": len(values),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "parameters": parameters,
    }


def gradcam_function(
    model: nn.Module, target_layer: nn.Conv2d, input_tensor: torch.Tensor
) -> tuple[Callable[[], np.ndarray], torch.utils.hooks.RemovableHandle]:
    activations: list[torch.Tensor] = []
    handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )

    def run() -> np.ndarray:
        activations.clear()
        model.zero_grad(set_to_none=True)
        inputs = input_tensor.detach().clone().requires_grad_(True)
        logits = model(inputs)
        if len(activations) != 1:
            raise RuntimeError("Captura Grad-CAM inesperada durante la latencia.")
        target = int(logits[0].argmax().item())
        gradient = torch.autograd.grad(logits[0, target], activations[0])[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        raw = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        resized = torch.nn.functional.interpolate(
            raw, size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        maximum = resized.max()
        minimum = resized.min()
        normalized = torch.where(
            maximum - minimum > 1e-12,
            (resized - minimum) / (maximum - minimum),
            torch.zeros_like(resized),
        )
        return normalized.detach().cpu().numpy()

    return run, handle


def verify_representatives(config: dict) -> list[dict[str, object]]:
    comparison = pd.read_csv(ROOT / "reports/main_experiments/comparison_nine_runs.csv")
    records = []
    for architecture, expected in config["latency"]["architectures"].items():
        subset = comparison[comparison["architecture"] == architecture].copy()
        median = float(subset["f1_macro"].median())
        subset["distance"] = (subset["f1_macro"] - median).abs()
        chosen = subset.sort_values(["distance", "seed"]).iloc[0]
        valid = (
            int(chosen["seed"]) == int(expected["seed"])
            and int(chosen["best_epoch"]) == int(expected["epoch"])
            and abs(float(chosen["f1_macro"]) - float(expected["selection_f1_macro"])) < 1e-12
        )
        if not valid:
            raise RuntimeError(f"Checkpoint representativo incompatible para {architecture}.")
        records.append(
            {
                "architecture": architecture,
                "median_selection_f1_macro": median,
                "representative_seed": int(chosen["seed"]),
                "representative_epoch": int(chosen["best_epoch"]),
                "representative_selection_f1_macro": float(chosen["f1_macro"]),
                "selection_only": True,
                "test_performance_consulted": False,
            }
        )
    return records


def main() -> None:
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    latency = config["latency"]
    warmups = int(latency["warmup_iterations"])
    measurements = int(latency["measured_iterations"])
    if (warmups, measurements) != (20, 100):
        raise RuntimeError("El protocolo exige 20 calentamientos y 100 mediciones.")
    if not torch.cuda.is_available():
        raise RuntimeError("La comparación de latencia requiere la RTX 3060 mediante CUDA.")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    representatives = verify_representatives(config)
    selected = pd.read_csv(ROOT / config["validation"]["selected_examples"]).sort_values("image_id")
    sample = selected.iloc[0]
    sample_path = ROOT / "data" / str(sample["relative_path"])
    payload = sample_path.read_bytes()
    output = OUTPUT_ROOT / f"representative_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    def preprocess_step() -> torch.Tensor:
        image = decode_image(payload)
        tensor, _ = preprocess_image(image)
        return tensor

    preprocessing_values = timed_iterations(preprocess_step, warmups, measurements, torch.device("cpu"))
    summary_rows.append(summarize("shared", "decode_and_preprocess", 1, preprocessing_values, None))
    raw_rows.extend(
        {"architecture": "shared", "component": "decode_and_preprocess", "batch_shape": 1, "iteration": index + 1, "milliseconds": value}
        for index, value in enumerate(preprocessing_values)
    )
    tensor, display = preprocess_image(decode_image(payload))
    model_load_rows = []
    for architecture, specification in latency["architectures"].items():
        checkpoint = ROOT / specification["checkpoint"]
        synchronize(device)
        started = time.perf_counter()
        model = load_benchmark_model(
            architecture,
            checkpoint,
            specification["checkpoint_sha256"],
            int(specification["seed"]),
            int(specification["epoch"]),
            device,
        )
        synchronize(device)
        load_seconds = time.perf_counter() - started
        parameters = sum(parameter.numel() for parameter in model.parameters())
        target_layer = dict(model.named_modules()).get(specification["target_layer"])
        if not isinstance(target_layer, nn.Conv2d):
            raise RuntimeError(f"Capa Grad-CAM inválida para {architecture}.")
        model_load_rows.append(
            {
                "architecture": architecture,
                "checkpoint": specification["checkpoint"],
                "checkpoint_sha256": sha256(checkpoint),
                "seed": specification["seed"],
                "epoch": specification["epoch"],
                "selection_f1_macro": specification["selection_f1_macro"],
                "parameters": parameters,
                "model_initialization_and_checkpoint_load_seconds": load_seconds,
                "target_layer": specification["target_layer"],
            }
        )
        input_one = tensor.to(device)
        torch.cuda.reset_peak_memory_stats(device)

        def inference_one() -> torch.Tensor:
            with torch.inference_mode():
                return model(input_one)

        inference_values = timed_iterations(inference_one, warmups, measurements, device)
        summary_rows.append(summarize(architecture, "inference", 1, inference_values, parameters))
        raw_rows.extend(
            {"architecture": architecture, "component": "inference", "batch_shape": 1, "iteration": index + 1, "milliseconds": value}
            for index, value in enumerate(inference_values)
        )
        gradcam_one, handle = gradcam_function(model, target_layer, input_one)
        try:
            gradcam_values = timed_iterations(gradcam_one, warmups, measurements, device)
        finally:
            handle.remove()
        summary_rows.append(summarize(architecture, "gradcam", 1, gradcam_values, parameters))
        raw_rows.extend(
            {"architecture": architecture, "component": "gradcam", "batch_shape": 1, "iteration": index + 1, "milliseconds": value}
            for index, value in enumerate(gradcam_values)
        )
        if architecture == "efficientnet_b0":
            input_demo = tensor.repeat(int(config["inference"]["forward_batch_shape"]), 1, 1, 1).to(device)

            def inference_demo() -> torch.Tensor:
                with torch.inference_mode():
                    return model(input_demo)

            demo_inference_values = timed_iterations(inference_demo, warmups, measurements, device)
            summary_rows.append(summarize(architecture, "demonstrator_inference", 32, demo_inference_values, parameters))
            raw_rows.extend(
                {"architecture": architecture, "component": "demonstrator_inference", "batch_shape": 32, "iteration": index + 1, "milliseconds": value}
                for index, value in enumerate(demo_inference_values)
            )
            gradcam_demo, demo_handle = gradcam_function(model, target_layer, input_demo)
            try:
                demo_gradcam_values = timed_iterations(gradcam_demo, warmups, measurements, device)
            finally:
                demo_handle.remove()
            summary_rows.append(summarize(architecture, "demonstrator_gradcam", 32, demo_gradcam_values, parameters))
            raw_rows.extend(
                {"architecture": architecture, "component": "demonstrator_gradcam", "batch_shape": 32, "iteration": index + 1, "milliseconds": value}
                for index, value in enumerate(demo_gradcam_values)
            )
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        model_load_rows[-1]["peak_allocated_mib_during_measurements"] = peak_mib
        del model, input_one
        gc.collect()
        torch.cuda.empty_cache()
    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    loads = pd.DataFrame(model_load_rows)
    representatives_frame = pd.DataFrame(representatives)
    raw.to_csv(output / "latency_raw_100_measurements.csv", index=False)
    summary.to_csv(output / "latency_summary.csv", index=False)
    loads.to_csv(output / "representative_checkpoints.csv", index=False)
    representatives_frame.to_csv(output / "representative_selection_verification.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows_markdown = []
    for row in summary.to_dict("records"):
        rows_markdown.append(
            f"| {row['architecture']} | {row['component']} | {row['batch_shape']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    report = "\n".join(
        [
            "# Comparación de latencia de inferencia",
            "",
            "Los checkpoints representativos se eligieron exclusivamente por la regla congelada de selección: F1 macro más próximo a la mediana de cada arquitectura. No se consultó el rendimiento de prueba para elegirlos.",
            "",
            "Cada componente cronometrado recibió 20 pasadas de calentamiento y 100 mediciones. Las mediciones CUDA se sincronizaron antes y después de cada pasada. El lote uno es la comparación principal del protocolo. El preprocesamiento incluye lectura desde bytes, validación, decodificación, conversión RGB, redimensionado, tensor y normalización.",
            "",
            "| Arquitectura | Componente | Forma de lote | Mediana ms | P95 ms |",
            "| --- | --- | ---: | ---: | ---: |",
            *rows_markdown,
            "",
            "La ruta real del demostrador repite una imagen 32 veces para reproducir las probabilidades del evaluador final dentro de 0,00001. Sus filas `demonstrator_inference` y `demonstrator_gradcam` representan el coste real por solicitud; no sustituyen la comparación principal con lote uno.",
            "",
            f"Imagen fija utilizada: `{sample['image_id']}`. GPU: `{torch.cuda.get_device_name(device)}`. No se ejecutó entrenamiento, recalibración ni inferencia completa del conjunto de prueba.",
        ]
    )
    (output / "INFORME_LATENCIAS.md").write_text(report + "\n", encoding="utf-8")
    technical = {
        "completed_at_utc": utc_now(),
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "warmup_iterations_per_component": warmups,
        "measured_iterations_per_component": measurements,
        "cuda_synchronized": True,
        "sample_image_id": str(sample["image_id"]),
        "sample_sha256": sha256(sample_path),
        "representative_rule_verified_from_selection": True,
        "test_performance_consulted_for_checkpoint_choice": False,
        "full_test_inference_repeated": False,
        "raw_measurement_rows": len(raw),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "benchmark_script_sha256": sha256(Path(__file__)),
    }
    (output / "technical_checks.json").write_text(json.dumps(technical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary.to_dict("records"), **technical}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
