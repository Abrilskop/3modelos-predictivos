import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Precios - California Housing", layout="wide")

st.title("Simulador de Precios de Viviendas en California")
st.markdown("Modelo de **Regresión Lineal Múltiple** entrenado con el dataset California Housing (Census 1990).")


@st.cache_data
def load_and_process_data():
    df = pd.read_csv("housing.csv")

    df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
    df["ocean_proximity_near"] = df["ocean_proximity"].apply(
        lambda x: 1 if x in ["NEAR OCEAN", "NEAR BAY"] else 0
    )

    features = [
        "housing_median_age",
        "rooms_per_household",
        "bedrooms_per_household",
        "ocean_proximity_near",
        "median_income",
        "longitude",
        "latitude",
    ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + ["median_house_value"], inplace=True)

    X = df[features].copy()
    y = df["median_house_value"].copy()

    return X, y, df, features


X, y, df_full, feature_names = load_and_process_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.header("Metricas del Modelo (Conjunto de Prueba)")
col1, col2, col3 = st.columns(3)
col1.metric("R2 Score", f"{r2:.4f}", help="Proporcion de varianza explicada")
col2.metric("MAE", f"${mae:,.0f}", help="Error Absoluto Medio en USD")
col3.metric("RMSE", f"${rmse:,.0f}", help="Raiz del Error Cuadratico Medio en USD")

st.header("Grafico: Valor Real vs Predicho")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Prediccion perfecta")
ax.set_xlabel("Valor Real (USD)")
ax.set_ylabel("Valor Predicho (USD)")
ax.set_title("Real vs Predicho")
ax.legend()
st.pyplot(fig)

st.header("Prediccion Interactiva")
st.markdown("Ajusta los parametros para estimar el precio de una vivienda.")

col_sl1, col_sl2 = st.columns(2)

with col_sl1:
    housing_age = st.slider(
        "Antiguedad de la casa (anios)",
        min_value=1, max_value=52, value=28, step=1,
    )
    rooms_per_hh = st.slider(
        "Cuartos por hogar",
        min_value=0.8, max_value=30.0, value=5.4, step=0.1,
    )
    bedrooms_per_hh = st.slider(
        "Dormitorios por hogar",
        min_value=0.3, max_value=10.0, value=1.1, step=0.1,
    )
    ocean_near = st.selectbox(
        "Cercania al mar",
        options=[("No", 0), ("Si (NEAR OCEAN / NEAR BAY)", 1)],
        format_func=lambda x: x[0],
    )

with col_sl2:
    median_income = st.slider(
        "Ingreso medio del bloque (x10K USD)",
        min_value=0.5, max_value=15.0, value=3.8, step=0.1,
    )
    longitude = st.slider(
        "Longitud",
        min_value=-124.3, max_value=-114.3, value=-119.6, step=0.1,
    )
    latitude = st.slider(
        "Latitud",
        min_value=32.5, max_value=42.0, value=35.6, step=0.1,
    )

input_data = pd.DataFrame([{
    "housing_median_age": housing_age,
    "rooms_per_household": rooms_per_hh,
    "bedrooms_per_household": bedrooms_per_hh,
    "ocean_proximity_near": ocean_near[1],
    "median_income": median_income,
    "longitude": longitude,
    "latitude": latitude,
}])

prediction = model.predict(input_data)[0]

st.divider()
st.subheader("Precio Estimado")
st.metric("Precio de la vivienda", f"${prediction:,.0f} USD")

st.divider()
with st.expander("Coeficientes del Modelo"):
    coef_df = pd.DataFrame({
        "Descriptor": [
            "Antiguedad (anios)",
            "Cuartos/hogar",
            "Dormitorios/hogar",
            "Cercania al mar",
            "Ingreso medio",
            "Longitud",
            "Latitud",
        ],
        "Coeficiente": model.coef_,
    })
    st.dataframe(coef_df, use_container_width=True)
    st.caption(f"Intercepto: ${model.intercept_:,.0f}")
