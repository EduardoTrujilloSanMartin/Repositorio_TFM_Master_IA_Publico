#!/usr/bin/env python3
"""Demostrador académico local de clasificación de lesiones cutáneas."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inference_core import InferenceEngine, InvalidImageError  # noqa: E402


st.set_page_config(page_title="Demostrador académico HAM10000", layout="wide")
st.title("Demostrador académico de lesiones cutáneas")
st.error(
    "Prototipo académico sin validación clínica. No proporciona diagnósticos, "
    "no sustituye la valoración profesional y no garantiza detectar imágenes fuera del dominio."
)
st.caption(
    "Sistema congelado: EfficientNetB0 ponderada, semilla 43, época 21; "
    "temperatura 2,0841459914047333 y umbral 0,5080122603583038."
)


@st.cache_resource
def load_engine() -> InferenceEngine:
    return InferenceEngine()


uploaded = st.file_uploader("Cargue una imagen dermatoscópica JPEG o PNG", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        payload = uploaded.getvalue()
        engine = load_engine()
        result = engine.predict(payload, with_gradcam=True)
        left, right = st.columns(2)
        with left:
            st.image(payload, caption="Imagen cargada", width="stretch")
        with right:
            st.image(result.overlay, caption="Grad-CAM de la clase predicha · opacidad 0,4", width="stretch")
        state = "ACEPTADA" if result.accepted else "ABSTENCIÓN"
        st.subheader(f"Clase predicha: {result.predicted_class}")
        st.write(f"Confianza calibrada: **{result.confidence:.6f}**")
        if result.accepted:
            st.success(f"{state}: confianza ≥ umbral congelado")
        else:
            st.warning(f"{state}: confianza < umbral congelado")
        table = pd.DataFrame(
            {"clase": engine.class_order, "probabilidad calibrada": result.probabilities}
        ).sort_values("probabilidad calibrada", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")
        st.bar_chart(table.set_index("clase"))
        if result.null_map:
            st.warning("El mapa Grad-CAM resultó nulo y se muestra explícitamente como tal.")
        st.caption(
            "Grad-CAM es una visualización de sensibilidad del modelo; no demuestra razonamiento clínico. "
            "La abstención es una regla experimental de confianza y no un detector validado de imágenes fuera del dominio."
        )
    except InvalidImageError as error:
        st.error(f"Archivo inválido: {error}")
    except Exception as error:
        st.error(f"No se pudo procesar la imagen: {error}")
