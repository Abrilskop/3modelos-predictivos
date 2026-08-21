<<<<<<< HEAD
# 3modelos-predictivos
=======
# Caso 3: Diabetes — Modelos Predictivos
>>>>>>> 6212d8f (feat: agregar caso3 diabetes polinomial)

**Modelado Predictivo Multisectorial con Regresion Lineal Multiple, Polinomial y Despliegue de Aplicativos de IA**

<<<<<<< HEAD
Asignatura: Inteligencia Artificial / Aprendizaje Automatico
Semestre: 2026-II

## Estructura del Proyecto

```
3modelos-predictivos/
├── requirements.txt              ← Dependencias del proyecto
├── app.py                        ← Aplicacion principal Streamlit (Fase C)
├── caso1_california/             ← Caso 1: California Housing
├── caso2_wine/                   ← Caso 2: Wine Quality
│   ├── modelo_wine.py            ← Entrenamiento (Fase A + B)
│   ├── modelo_wine.pkl           ← Modelo serializado
│   ├── app_wine.py               ← App Streamlit individual
│   ├── matriz_correlacion.png    ← Matriz de correlacion
│   ├── comparativa_modelos.png   ← Comparativa de modelos
│   └── comparativa_r2.png        ← Grafico R²
├── caso3_diabetes/               ← Caso 3: Diabetes (Scikit-learn)
├── LICENSE
└── README.md
```
=======
Predice la **progresión cuantitativa de la diabetes** con modelos de aprendizaje supervisado (Regresión Lineal Múltiple y Regresión Polinomial de Grado 2) sobre el dataset de scikit-learn.

## Modelos y Archivos

### 1. Regresión Lineal Convencional (`caso3_diabetes`)
*   **`modelo_diabetes.py`**: Entrenamiento del modelo lineal básico y predicciones por consola.
*   **`app_diabetes.py`**: Interfaz web interactiva con Streamlit.

### 2. Regresión Polinomial Grado 2 (`caso3_diabetes_polinomial`)
*   **`modelo_diabetes_polinomial.py`**: Pipeline de entrenamiento con `PolynomialFeatures` (generando 65 características polinomiales) y regresión lineal, con visualización de coeficientes e interacción de consola.
*   **`app_diabetes_polinomial.py`**: Interfaz interactiva de Streamlit para predicciones polinomiales en tiempo real y explicación de sobreajuste (overfitting).

---
>>>>>>> 6212d8f (feat: agregar caso3 diabetes polinomial)

## Casos de Estudio

| Caso | Dataset | Objetivo | Variables |
|------|---------|----------|-----------|
| 1 | California Housing | Predecir valor de vivienda | 8 features |
| 2 | Wine Quality | Predecir calidad de vino (0-10) | 6 features |
| 3 | Diabetes | Predecir progresion de enfermedad | 10 features |

## Caso 2: Wine Quality (Completado)

### Variables Seleccionadas (6 de 11)
1. `alcohol` — Predictor #1, correlacion fuerte positiva
2. `volatile_acidity` — Predictor #2, correlacion negativa
3. `sulphates` — Correlacion positiva moderada
4. `citric_acid` — Senal de frescura
5. `density` — Balance azucar/alcohol
6. `chlorides` — Contenido de sal

### Fases Implementadas
- **Fase A:** EDA + Preprocesamiento (limpieza, outliers IQR, VIF, StandardScaler)
- **Fase B:** Modelos de Regresion Lineal + Polinomial (grados 2 y 3)
- **Fase C:** App Streamlit con botella de vino animada

### Ejecucion

```bash
# Instalar dependencias
pip install -r requirements.txt

<<<<<<< HEAD
# Entrenar modelo (genera modelo_wine.pkl)
python caso2_wine/modelo_wine.py

# Ejecutar app individual
streamlit run caso2_wine/app_wine.py
=======
# --- REGRESIÓN LINEAL CONVENCIONAL ---
# Modo consola
python caso3_diabetes/modelo_diabetes.py
# Interfaz Streamlit
streamlit run caso3_diabetes/app_diabetes.py

# --- REGRESIÓN POLINOMIAL GRADO 2 ---
# Modo consola
python caso3_diabetes_polinomial/modelo_diabetes_polinomial.py
# Interfaz Streamlit
streamlit run caso3_diabetes_polinomial/app_diabetes_polinomial.py
>>>>>>> 6212d8f (feat: agregar caso3 diabetes polinomial)
```

## Tecnologias

<<<<<<< HEAD
- Python 3.10+
- Scikit-learn
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn
- Statsmodels (VIF)

## Licencia

MIT
=======
```powershell
cd "d:\IA"
git clone https://github.com/Abrilskop/3modelos-predictivos.git
# Copiar modelo de regresión lineal convencional
Copy-Item -Recurse "d:\IA\practica modelado\caso3_diabetes" "d:\IA\3modelos-predictivos\"
# Copiar modelo de regresión polinomial
Copy-Item -Recurse "d:\IA\practica modelado\caso3_diabetes_polinomial" "d:\IA\3modelos-predictivos\"
cd 3modelos-predictivos
git add caso3_diabetes caso3_diabetes_polinomial
git commit -m "Agregar Caso 3: Diabetes con regresión lineal y polinomial e interfaces Streamlit"
git push origin main
```
>>>>>>> 6212d8f (feat: agregar caso3 diabetes polinomial)
