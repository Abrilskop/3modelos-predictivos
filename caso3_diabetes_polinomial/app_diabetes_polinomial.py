import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelo_diabetes_polinomial import (
    FEATURE_LABELS,
    entrenar_modelo,
    obtener_rangos,
    predecir_progresion,
)

st.set_page_config(
    page_title="Predicción de Diabetes - Polinomial",
    page_icon="🩺",
    layout="centered",
)

st.title("Predicción de la progresión de la diabetes (Polinomial)")
st.markdown(
    "Ajuste los parámetros clínicos con las barras deslizantes para obtener "
    "una predicción en tiempo real basada en **regresión polinomial de grado 2**."
)


@st.cache_resource
def cargar_modelo():
    return entrenar_modelo()


model, X, metricas = cargar_modelo()
rangos = obtener_rangos(X)

with st.sidebar:
    st.header("Métricas del modelo polinomial")
    st.metric("R² (Grado 2)", f"{metricas['r2']:.4f}")
    st.metric("MSE (Grado 2)", f"{metricas['mse']:.2f}")
    st.caption("Entrenado con el dataset de diabetes de scikit-learn.")
    
    st.divider()
    st.subheader("Análisis del Modelo")
    st.info(
        "Al generar **65 características polinomiales** a partir de las 10 "
        "originales para solo 353 muestras de entrenamiento, la regresión lineal convencional "
        "presenta un leve sobreajuste (overfitting). "
        "Esto explica por qué el $R^2$ de prueba disminuye ligeramente en comparación con el "
        "modelo lineal convencional ($R^2 \\approx 0.4526$)."
    )

st.subheader("Parámetros clínicos")

valores = []
cols = st.columns(2)

for i, col in enumerate(X.columns):
    rango = rangos[col]
    etiqueta = FEATURE_LABELS.get(col, col)
    step = 1.0 if col == "sex" else 0.1

    with cols[i % 2]:
        valor = st.slider(
            etiqueta,
            min_value=rango["min"],
            max_value=rango["max"],
            value=rango["default"],
            step=step,
            key=col,
        )
        valores.append(valor)

prediccion = predecir_progresion(model, valores)

st.divider()
st.subheader("Resultado")
st.metric(
    label="Progreso cuantitativo predicho de la enfermedad",
    value=f"{prediccion:.2f}",
    help="Valor predicho por el modelo de regresión polinomial.",
)

st.info(
    "La predicción se actualiza automáticamente al mover cualquier barra. "
    "Los rangos corresponden a los valores mínimo y máximo del dataset de entrenamiento."
)
