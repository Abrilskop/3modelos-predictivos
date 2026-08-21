import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="Simulador de Precios Polinomial - California Housing", layout="wide")

st.title("Simulador de Precios de Viviendas en California")
st.markdown(
    "Modelo de **Regresión Polinomial** entrenado con el dataset California Housing (Census 1990). "
    "Selecciona el grado del polinomio para ajustar la complejidad del modelo."
)


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

st.sidebar.header("Configuración del Modelo")
degree = st.sidebar.slider(
    "Grado del polinomio",
    min_value=2,
    max_value=4,
    value=2,
    step=1,
    help="Mayor grado = modelo más complejo. Grado 2 es el recomendado."
)

alpha = st.sidebar.slider(
    "Alpha (regularización Ridge)",
    min_value=0.01,
    max_value=100.0,
    value=1.0,
    step=0.1,
    help="Controla la regularización para evitar overfitting."
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = Ridge(alpha=alpha)
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.header("Métricas del Modelo (Conjunto de Prueba)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Grado Polynomial", f"{degree}")
col2.metric("R² Score", f"{r2:.4f}", help="Proporción de varianza explicada")
col3.metric("MAE", f"${mae:,.0f}", help="Error Absoluto Medio en USD")
col4.metric("RMSE", f"${rmse:,.0f}", help="Raíz del Error Cuadrático Medio en USD")

st.divider()

st.header("Predicción Interactiva")
st.markdown("Ajusta los parámetros para estimar el precio de una vivienda.")

col_sl1, col_sl2 = st.columns(2)

with col_sl1:
    housing_age = st.slider(
        "Antigüedad de la casa (años)",
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
        "Cercanía al mar",
        options=[("No", 0), ("Sí (NEAR OCEAN / NEAR BAY)", 1)],
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

input_scaled = scaler.transform(input_data)
input_poly = poly.transform(input_scaled)
prediction = model.predict(input_poly)[0]

st.divider()

col_price, col_map = st.columns([1, 1])

with col_price:
    st.subheader("Precio Estimado")
    st.metric("Precio de la vivienda", f"${prediction:,.0f} USD")

with col_map:
    st.subheader("Ubicación Geográfica")
    map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_data, zoom=6)

st.divider()

st.header("Gráfico: Valor Real vs Predicho")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="darkorange")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predicción perfecta")


def dollars_formatter(x, pos):
    return f"${x:,.0f}"


ax.xaxis.set_major_formatter(FuncFormatter(dollars_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(dollars_formatter))
ax.set_xlabel("Valor Real (USD)")
ax.set_ylabel("Valor Predicho (USD)")
ax.set_title("Real vs Predicho (Regresión Polinomial)")
ax.legend()
st.pyplot(fig)

st.caption(
    "Nota: Se observa una concentración de valores en $500,000 USD debido al tope (cap) "
    "aplicado por el censo original de California de 1990, que limitó el valor máximo "
    "reportado de las viviendas a esta cifra."
)

st.divider()

with st.expander("Información del Modelo Polinomial"):
    st.write(f"**Grado del polinomio:** {degree}")
    st.write(f"**Alpha (regularización Ridge):** {alpha}")
    st.write(f"**Número de features originales:** {len(feature_names)}")
    st.write(f"**Número de features polinomiales:** {X_train_poly.shape[1]}")
    st.caption(f"Intercepto: ${model.intercept_:,.0f}")
