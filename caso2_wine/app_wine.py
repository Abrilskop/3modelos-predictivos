import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    data = joblib.load(os.path.join(BASE_DIR, 'modelo_wine.pkl'))
    return data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
degree = model_data['degree']
model_name = model_data['model_name']
metrics = model_data['metrics']

# --- Configuracion de pagina ---
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personalizado con scroll dinamico y botella ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

    /* Scroll suave */
    html {
        scroll-behavior: smooth;
    }

    /* Animaciones fade-in */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fillBottle {
        from { height: 0%; }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .fade-in-up { animation: fadeInUp 0.8s ease-out; }
    .fade-in-left { animation: fadeInLeft 0.8s ease-out; }
    .fade-in-right { animation: fadeInRight 0.8s ease-out; }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #2d1b33 0%, #5c2d4a 50%, #8b3a62 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: 2px;
    }
    .main-header p {
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }

    /* Secciones */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #8b3a62;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .section-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    /* Metricas */
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-box:hover { transform: scale(1.03); }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8b3a62;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }

    /* Botella de vino */
    .bottle-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem 0;
    }
    .bottle-wrapper {
        position: relative;
        width: 120px;
        height: 340px;
    }
    .bottle-svg {
        position: relative;
        z-index: 2;
    }
    .liquid {
        position: absolute;
        bottom: 52px;
        left: 50%;
        transform: translateX(-50%);
        width: 56px;
        border-radius: 0 0 12px 12px;
        transition: height 1.2s cubic-bezier(0.4, 0, 0.2, 1),
                    background 0.8s ease;
        z-index: 1;
        overflow: hidden;
    }
    .liquid::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 8px;
        background: rgba(255,255,255,0.3);
        border-radius: 50%;
    }
    .liquid-shimmer {
        position: absolute;
        top: 0;
        left: -50%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: shimmer 2s infinite;
    }

    .quality-label {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 1rem;
        transition: color 0.5s ease;
    }
    .quality-text {
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }

    /* Resultado prediction */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.5s ease;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b33 0%, #3d2645 100%);
    }
    [data-testid="stSidebar"] label {
        color: #e0d0e8 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #e0d0e8 !important;
    }

    /* Boton predecir */
    .stButton > button {
        background: linear-gradient(135deg, #8b3a62, #5c2d4a) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(139,58,98,0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139,58,98,0.6) !important;
    }

    /* Divider */
    .section-divider {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b3a62, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🍷 Wine Quality Predictor</h1>
    <p>Regresion Lineal Multiple y Polinomial — Analisis de Calidad de Vino</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: Inputs ---
st.sidebar.markdown("## 🍇 Parametros del Vino")
st.sidebar.markdown("Ajusta las propiedades fisicoquimicas:")

input_values = {}
input_values['alcohol'] = st.sidebar.slider(
    "🍷 Alcohol (% vol)", 8.0, 15.0, 10.5, 0.1,
    help="Contenido de alcohol del vino"
)
input_values['volatile acidity'] = st.sidebar.slider(
    "🧪 Acidez Volatil (g/dm³)", 0.08, 1.6, 0.52, 0.01,
    help="Cantidad de acido acetico. Valores altos dan sabor a vinagre"
)
input_values['sulphates'] = (
    st.sidebar.slider(
    "⚗️ Sulphatos (g/dm³)", 0.33, 2.0, 0.62, 0.01,
    help="Contribuyen al SO2, actuan como antimicrobiano"
)
)
input_values['citric acid'] = st.sidebar.slider(
    "🍋 Acido Citrico (g/dm³)", 0.0, 1.66, 0.27, 0.01,
    help="Aporta frescura y sabor al vino"
)
input_values['density'] = st.sidebar.slider(
    "⚖️ Densidad (g/cm³)", 0.987, 1.039, 0.997, 0.001,
    help="Relacionada con el contenido de azucar y alcohol"
)
input_values['chlorides'] = st.sidebar.slider(
    "🧂 Cloruros (g/dm³)", 0.009, 0.611, 0.056, 0.001,
    help="Contenido de sal en el vino"
)

# --- Funcion de prediccion ---
def predict_quality(values):
    X_input = pd.DataFrame([[
        values['alcohol'],
        values['volatile acidity'],
        values['sulphates'],
        values['citric acid'],
        values['density'],
        values['chlorides']
    ]], columns=features)
    X_scaled = scaler.transform(X_input)
    if degree > 1:
        poly = model_data['poly']
        X_poly = poly.transform(X_scaled)
        prediction = model.predict(X_poly)[0]
    else:
        prediction = model.predict(X_scaled)[0]
    return np.clip(prediction, 0, 10)

# --- Funcion para color de calidad ---
def get_quality_color(score):
    if score <= 3:
        return '#8B0000', 'Muy Baja', '#ffe0e0'
    elif score <= 4:
        return '#C41E3A', 'Baja', '#ffe8e8'
    elif score <= 5:
        return '#D2691E', 'Regular', '#fff3e0'
    elif score <= 6:
        return '#DAA520', 'Aceptable', '#fff8e1'
    elif score <= 7:
        return '#228B22', 'Buena', '#e8f5e9'
    elif score <= 8:
        return '#1B5E20', 'Muy Buena', '#e0f2e0'
    else:
        return '#FFD700', 'Excelente', '#fffde7'

# --- Funcion para generar botella SVG ---
def generate_bottle_html(score, color, label):
    fill_percent = min(score / 10.0 * 100, 100)
    fill_height = int(fill_percent * 2.5)

    bottle_html = f"""
    <div class="bottle-container">
        <div class="bottle-wrapper">
            <svg class="bottle-svg" width="120" height="340" viewBox="0 0 120 340">
                <!-- Sello superior -->
                <rect x="44" y="0" width="32" height="15" rx="3" fill="#8B4513"/>
                <rect x="46" y="5" width="28" height="8" rx="2" fill="#A0522D"/>

                <!-- Cuello de la botella -->
                <path d="M 48 15 L 48 80 Q 48 100 38 120 L 38 120" fill="none" stroke="#1a1a2e" stroke-width="4"/>
                <path d="M 72 15 L 72 80 Q 72 100 82 120 L 82 120" fill="none" stroke="#1a1a2e" stroke-width="4"/>
                <rect x="48" y="10" width="24" height="75" rx="2" fill="rgba(100,200,100,0.15)" stroke="#1a1a2e" stroke-width="3"/>

                <!-- Cuerpo de la botella -->
                <path d="M 38 120 Q 15 140 15 160 L 15 300 Q 15 325 40 325 L 80 325 Q 105 325 105 300 L 105 160 Q 105 140 82 120 Z"
                      fill="rgba(100,200,100,0.12)" stroke="#1a1a2e" stroke-width="3"/>

                <!-- Etiqueta -->
                <rect x="25" y="200" width="70" height="70" rx="5" fill="white" stroke="#ccc" stroke-width="1"/>
                <text x="60" y="225" text-anchor="middle" font-family="Playfair Display, serif" font-size="11" font-weight="700" fill="#2d1b33">WINE</text>
                <text x="60" y="242" text-anchor="middle" font-family="Lato, sans-serif" font-size="8" fill="#666">QUALITY</text>
                <line x1="35" y1="250" x2="85" y2="250" stroke="#8b3a62" stroke-width="1"/>
                <text x="60" y="264" text-anchor="middle" font-family="Playfair Display, serif" font-size="14" font-weight="700" fill="{color}">{score:.1f}</text>

                <!-- Base -->
                <ellipse cx="60" cy="325" rx="45" ry="8" fill="rgba(100,200,100,0.08)" stroke="#1a1a2e" stroke-width="2"/>
            </svg>

            <!-- Licor interior -->
            <div class="liquid" style="height: {fill_height}px; background: linear-gradient(180deg, {color}dd, {color}99);">
                <div class="liquid-shimmer"></div>
            </div>
        </div>

        <div class="quality-label" style="color: {color};">{score:.1f} / 10</div>
        <div class="quality-text">Calidad: {label}</div>
    </div>
    """
    return bottle_html

# --- CONTENIDO PRINCIPAL ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
    <div class="section-card fade-in-left">
        <h2 style="font-family: 'Playfair Display', serif; color: #2d1b33;">🔬 Fase A: Analisis Exploratorio</h2>
        <p style="color: #6c757d;">Preprocesamiento del dataset Wine Quality (UCI)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Variables Seleccionadas (6 de 11):**")
    st.markdown(
"| Variable | Descripcion | Importancia |\n"
"|----------|-------------|-------------|\n"
"| `alcohol` | Contenido de alcohol (% vol) | ⭐⭐⭐⭐⭐ |\n"
"| `volatile acidity` | Acidez volatil (g/dm³) | ⭐⭐⭐⭐ |\n"
"| `sulphates` | Sulphatos (g/dm³) | ⭐⭐⭐ |\n"
"| `citric acid` | Acido citrico (g/dm³) | ⭐⭐⭐ |\n"
"| `density` | Densidad (g/cm³) | ⭐⭐⭐ |\n"
"| `chlorides` | Cloruros/sal (g/dm³) | ⭐⭐ |"
    )

    st.markdown("**Preprocesamiento:**")
    st.markdown(
"- ✅ Sin valores nulos\n"
"- ✅ Duplicados eliminados\n"
"- ✅ Outliers tratados (IQR)\n"
"- ✅ Escalado: StandardScaler\n"
"- ✅ Multicolinealidad verificada (VIF)"
    )

with col_right:
    prediction = predict_quality(input_values)
    color, label, bg_color = get_quality_color(prediction)

    st.markdown(f"""
    <div class="result-card fade-in-right" style="background: {bg_color}; border: 2px solid {color}33;">
        <h2 style="font-family: 'Playfair Display', serif; color: #2d1b33; margin-bottom: 0;">Prediccion de Calidad</h2>
    """, unsafe_allow_html=True)

    # Botella animada
    bottle_html = generate_bottle_html(prediction, color, label)
    st.markdown(bottle_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- MODELOS ---
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-card fade-in-up">
    <h2 style="font-family: 'Playfair Display', serif; color: #2d1b33;">📊 Fase B: Modelamiento Estadistico</h2>
    <p style="color: #6c757d;">Comparativa de modelos de regresion</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">R²</div>
        <div class="metric-label">Coeficiente de Determinacion</div>
    </div>
    """, unsafe_allow_html=True)
    st.metric("Lineal", f"{metrics['R2']:.4f}")

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">MAE</div>
        <div class="metric-label">Error Absoluto Medio</div>
    </div>
    """, unsafe_allow_html=True)
    st.metric("Lineal", f"{metrics['MAE']:.4f}")

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-value">RMSE</div>
        <div class="metric-label">Raiz del Error Cuadratico</div>
    </div>
    """, unsafe_allow_html=True)
    st.metric("Lineal", f"{metrics['RMSE']:.4f}")

st.info(f"**Modelo seleccionado:** {model_name} | **Grado polinomial:** {degree}")

# --- TABLA DE VALORES INGRESADOS ---
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-card fade-in-up">
    <h2 style="font-family: 'Playfair Display', serif; color: #2d1b33;">📋 Valores Ingresados</h2>
</div>
""", unsafe_allow_html=True)

df_input = pd.DataFrame([input_values])
st.dataframe(df_input.style.format({
    'alcohol': '{:.1f}',
    'volatile acidity': '{:.2f}',
    'sulphates': '{:.2f}',
    'citric acid': '{:.2f}',
    'density': '{:.3f}',
    'chlorides': '{:.3f}'
}).set_properties(**{'text-align': 'center'}), width="stretch")

# --- FOOTER ---
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #999; padding: 1rem; font-family: 'Lato', sans-serif;">
    <p>🍷 Wine Quality Predictor — Modelado Predictivo Multisectorial</p>
    <p style="font-size: 0.8rem;">Dataset: UCI Machine Learning Repository | Modelo: """ + model_name + """</p>
</div>
""", unsafe_allow_html=True)
