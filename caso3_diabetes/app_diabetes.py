import streamlit as st

from modelo_diabetes import FEATURE_LABELS, entrenar_modelo, obtener_rangos, predecir_progresion

st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered",
)

st.title("Predicción de la progresión de la diabetes")
st.markdown(
    "Ajuste los parámetros clínicos con las barras deslizantes para obtener "
    "una predicción en tiempo real basada en regresión lineal múltiple."
)


@st.cache_resource
def cargar_modelo():
    return entrenar_modelo()


model, X, metricas = cargar_modelo()
rangos = obtener_rangos(X)

with st.sidebar:
    st.header("Métricas del modelo")
    st.metric("R² (conjunto de prueba)", f"{metricas['r2']:.4f}")
    st.metric("MSE (conjunto de prueba)", f"{metricas['mse']:.2f}")
    st.caption("Entrenado con el dataset de diabetes de scikit-learn.")

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
    label="Progresión cuantitativa de la enfermedad",
    value=f"{prediccion:.2f}",
    help="Valor predicho por el modelo de regresión lineal.",
)

st.info(
    "La predicción se actualiza automáticamente al mover cualquier barra. "
    "Los rangos corresponden a los valores mínimo y máximo del dataset de entrenamiento."
)
