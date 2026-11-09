#!/usr/bin/env python3
"""Ruta compartida de inferencia del demostrador académico del TFM."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import matplotlib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_simple_cnn import CLASS_ORDER, SimpleCNN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/demonstrator_primary.json"
matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


class InvalidImageError(ValueError):
    """Archivo ausente, demasiado grande, no decodificable o de formato no admitido."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(architecture: str) -> nn.Module:
    if architecture == "cnn_simple":
        return SimpleCNN(num_classes=len(CLASS_ORDER))
    if architecture == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(weights=None)
    elif architecture == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
    else:
        raise ValueError(f"Arquitectura no reconocida: {architecture}")
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, len(CLASS_ORDER))
    return model


def decode_image(
    source: Path | bytes | bytearray | BinaryIO,
    *,
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG"),
    maximum_bytes: int = 25 * 1024 * 1024,
) -> Image.Image:
    if isinstance(source, Path):
        if not source.is_file():
            raise InvalidImageError("El archivo no existe.")
        payload = source.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
    elif hasattr(source, "read"):
        payload = source.read()
    else:
        raise InvalidImageError("Tipo de entrada no admitido.")
    if not payload:
        raise InvalidImageError("El archivo está vacío.")
    if len(payload) > maximum_bytes:
        raise InvalidImageError(f"El archivo supera el límite de {maximum_bytes} bytes.")
    try:
        with Image.open(io.BytesIO(payload)) as candidate:
            image_format = (candidate.format or "").upper()
            candidate.verify()
        if image_format not in allowed_formats:
            raise InvalidImageError(
                f"Formato {image_format or 'desconocido'} no admitido; use JPEG o PNG."
            )
        with Image.open(io.BytesIO(payload)) as candidate:
            candidate.load()
            return candidate.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("El archivo no es una imagen JPEG/PNG íntegra y decodificable.") from error


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    resized = TF.resize(
        image.convert("RGB"),
        [224, 224],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    display = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = TF.normalize(
        TF.to_tensor(resized),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0)
    return tensor, display


def decision_from_confidence(confidence: float, threshold: float) -> bool:
    return float(confidence) >= float(threshold)


@dataclass
class Prediction:
    logits: np.ndarray
    probabilities: np.ndarray
    predicted_index: int
    predicted_class: str
    confidence: float
    accepted: bool
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None
    null_map: bool | None = None
    map_finite: bool | None = None


class InferenceEngine:
    """Carga una vez el sistema congelado y ejecuta predicción y Grad-CAM."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG, device: str | None = None) -> None:
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        system = self.config["system"]
        self.class_order = list(system["class_order"])
        if self.class_order != CLASS_ORDER:
            raise RuntimeError("El orden de clases no coincide con el protocolo congelado.")
        self.temperature = float(system["temperature"])
        self.threshold = float(system["threshold"])
        self.default_batch_shape = int(self.config["inference"]["forward_batch_shape"])
        self.device = torch.device(device or self.config["inference"]["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible para la ruta congelada del demostrador.")
        self.checkpoint = ROOT / system["checkpoint"]
        if sha256(self.checkpoint) != system["checkpoint_sha256"]:
            raise RuntimeError("La huella del checkpoint principal no coincide.")
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "architecture": system["architecture"],
            "training_seed": system["training_seed"],
            "epoch": system["checkpoint_epoch"],
            "partition_seed": system["partition_seed"],
            "class_order": self.class_order,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"Metadato incompatible en el checkpoint: {key}")
        self.model = build_model(system["architecture"])
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.target_layer_name = self.config["gradcam"]["target_layer"]
        self.target_layer = dict(self.model.named_modules()).get(self.target_layer_name)
        if not isinstance(self.target_layer, nn.Conv2d):
            raise RuntimeError("La capa Grad-CAM congelada no es una Conv2d válida.")

    def decode_and_preprocess(self, source: Path | bytes | bytearray | BinaryIO) -> tuple[torch.Tensor, np.ndarray]:
        preprocessing = self.config["preprocessing"]
        image = decode_image(
            source,
            allowed_formats=tuple(preprocessing["accepted_file_formats"]),
            maximum_bytes=int(preprocessing["maximum_upload_bytes"]),
        )
        return preprocess_image(image)

    def _batch(self, tensor: torch.Tensor, batch_shape: int) -> torch.Tensor:
        if tensor.shape != (1, 3, 224, 224):
            raise ValueError(f"Tensor preprocesado inesperado: {tuple(tensor.shape)}")
        if batch_shape < 1:
            raise ValueError("La forma de lote debe ser positiva.")
        return tensor.repeat(batch_shape, 1, 1, 1).to(self.device)

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        display: np.ndarray | None = None,
        *,
        with_gradcam: bool = False,
        batch_shape: int | None = None,
    ) -> Prediction:
        size = int(batch_shape or self.default_batch_shape)
        batch = self._batch(tensor, size)
        if not with_gradcam:
            with torch.inference_mode():
                logits_all = self.model(batch)
            logits = logits_all[0]
            probabilities = torch.softmax(logits / self.temperature, dim=0)
            return self._prediction(logits, probabilities)

        if display is None or display.shape != (224, 224, 3):
            raise ValueError("Grad-CAM necesita la imagen RGB redimensionada de 224x224.")
        activations: list[torch.Tensor] = []
        hook = self.target_layer.register_forward_hook(
            lambda _module, _inputs, output: activations.append(output)
        )
        try:
            batch.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits_all = self.model(batch)
            if len(activations) != 1:
                raise RuntimeError("No se capturó exactamente una activación Grad-CAM.")
            logits = logits_all[0]
            probabilities = torch.softmax(logits / self.temperature, dim=0)
            predicted_index = int(logits.argmax().item())
            gradient = torch.autograd.grad(
                logits[predicted_index], activations[0], retain_graph=False, create_graph=False
            )[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            raw = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            raw = torch.nn.functional.interpolate(
                raw, size=(224, 224), mode="bilinear", align_corners=False
            )[0, 0].detach().cpu().numpy()
        finally:
            hook.remove()
        finite = bool(np.isfinite(raw).all())
        minimum = float(np.nanmin(raw)) if finite else 0.0
        maximum = float(np.nanmax(raw)) if finite else 0.0
        span = maximum - minimum
        null_map = (not finite) or span <= 1e-12 or maximum <= 1e-12
        heatmap = np.zeros_like(raw, dtype=np.float32) if null_map else ((raw - minimum) / span).astype(np.float32)
        colored = colormaps["jet"](heatmap)[..., :3].astype(np.float32)
        opacity = float(self.config["gradcam"]["overlay_opacity"])
        overlay = np.clip((1.0 - opacity) * display + opacity * colored, 0.0, 1.0)
        prediction = self._prediction(logits, probabilities)
        prediction.heatmap = heatmap
        prediction.overlay = overlay
        prediction.null_map = bool(null_map)
        prediction.map_finite = finite
        return prediction

    def predict(
        self,
        source: Path | bytes | bytearray | BinaryIO,
        *,
        with_gradcam: bool = False,
        batch_shape: int | None = None,
    ) -> Prediction:
        tensor, display = self.decode_and_preprocess(source)
        return self.predict_tensor(
            tensor, display, with_gradcam=with_gradcam, batch_shape=batch_shape
        )

    def _prediction(self, logits: torch.Tensor, probabilities: torch.Tensor) -> Prediction:
        predicted_index = int(logits.argmax().item())
        confidence = float(probabilities[predicted_index].detach().cpu().item())
        return Prediction(
            logits=logits.detach().cpu().numpy(),
            probabilities=probabilities.detach().cpu().numpy(),
            predicted_index=predicted_index,
            predicted_class=self.class_order[predicted_index],
            confidence=confidence,
            accepted=decision_from_confidence(confidence, self.threshold),
        )


def load_benchmark_model(
    architecture: str,
    checkpoint: Path,
    expected_sha256: str,
    expected_seed: int,
    expected_epoch: int,
    device: torch.device,
) -> nn.Module:
    if sha256(checkpoint) != expected_sha256:
        raise RuntimeError(f"Huella incompatible para {architecture}.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("epoch") != expected_epoch or state.get("class_order") != CLASS_ORDER:
        raise RuntimeError(f"Checkpoint representativo incompatible para {architecture}.")
    if architecture == "cnn_simple":
        stored_seed = state.get("config", {}).get("training", {}).get("seed")
    else:
        stored_seed = state.get("training_seed")
        if state.get("architecture") != architecture or state.get("partition_seed") != 2026:
            raise RuntimeError(f"Metadatos incompatibles para {architecture}.")
    if stored_seed != expected_seed:
        raise RuntimeError(f"Semilla incompatible para {architecture}.")
    model = build_model(architecture)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
