import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CASO 2: WINE QUALITY — FASE A + B")
print("Regresion Lineal Multiple y Polinomial")
print("=" * 70)

# ============================================================
# FASE A: ANALISIS EXPLORATORIO Y PREPROCESAMIENTO
# ============================================================
print("\n" + "=" * 70)
print("FASE A: ANALISIS EXPLORATORIO Y PREPROCESAMIENTO")
print("=" * 70)

# --- 1. Carga de datos ---
print("\n[1] Cargando datasets desde UCI...")
url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

red_wine = pd.read_csv(url_red, sep=';')
white_wine = pd.read_csv(url_white, sep=';')

red_wine['type'] = 0  # tinto
white_wine['type'] = 1  # blanco

wine = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

print(f"   Vinos tintos: {len(red_wine)} muestras")
print(f"   Vinos blancos: {len(white_wine)} muestras")
print(f"   Total combinado: {len(wine)} muestras")
print(f"   Columnas: {list(wine.columns)}")

# --- 2. Informacion general ---
print("\n[2] Informacion general del dataset:")
print(wine.info())

print("\n[3] Estadisticas descriptivas:")
print(wine.describe().round(3))

# --- 3. Valores nulos ---
print("\n[4] Valores nulos por columna:")
nulos = wine.isnull().sum()
print(nulos)
print(f"   Total nulos: {nulos.sum()}")

# --- 4. Eliminacion de duplicados ---
duplicados_antes = len(wine)
wine = wine.drop_duplicates()
duplicados_despues = len(wine)
print(f"\n[5] Duplicados eliminados: {duplicados_antes - duplicados_despues}")
print(f"   Muestras restantes: {duplicados_despues}")

# --- 5. Tratamiento de outliers (IQR) ---
print("\n[6] Tratamiento de outliers (metodo IQR):")
features_numericas = wine.drop(columns=['quality', 'type']).columns
outliers_total = 0

for col in features_numericas:
    Q1 = wine[col].quantile(0.25)
    Q3 = wine[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (wine[col] >= lower) & (wine[col] <= upper)
    outliers_col = (~mask).sum()
    outliers_total += outliers_col
    wine = wine[mask]

print(f"   Total outliers removidos: {outliers_total}")
print(f"   Muestras finales: {len(wine)}")

# --- 6. Seleccion de 6 variables ---
print("\n[7] Seleccion de 6 variables predictoras:")
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'density', 'chlorides']
print(f"   Variables: {selected_features}")

X = wine[selected_features]
y = wine['quality']

# --- 7. Analisis de correlacion ---
print("\n[8] Matriz de correlacion:")
corr_matrix = wine[selected_features + ['quality']].corr()
print(corr_matrix['quality'].drop('quality').sort_values(ascending=False).round(3))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
ax.set_title('Matriz de Correlacion — Wine Quality (6 Variables)')
plt.tight_layout()
plt.savefig('caso2_wine/matriz_correlacion.png', dpi=150)
plt.close()
print("   Guardado: caso2_wine/matriz_correlacion.png")

# --- 8. VIF (Multicolinealidad) ---
print("\n[9] Analisis VIF (Variance Inflation Factor):")
vif_data = pd.DataFrame()
vif_data['Variable'] = selected_features
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data.to_string(index=False))
print("   (VIF > 10 indica multicolinealidad alta)")

# --- 9. Escalado ---
print("\n[10] Escalado con StandardScaler:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
print("   Datos escalados correctamente")

# --- 10. Split train/test ---
print("\n[11] Division train/test (80/20):")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]} muestras")
print(f"   Test: {X_test.shape[0]} muestras")

# ============================================================
# FASE B: MODELAMIENTO ESTADISTICO
# ============================================================
print("\n" + "=" * 70)
print("FASE B: MODELAMIENTO ESTADISTICO")
print("=" * 70)

results = {}

# --- Modelo 1: Regresion Lineal Multiple ---
print("\n[12] Modelo 1: Regresion Lineal Multiple")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

results['Lineal Multiple'] = {'R2': r2_lr, 'MAE': mae_lr, 'RMSE': rmse_lr}

print(f"   R²:  {r2_lr:.4f}")
print(f"   MAE: {mae_lr:.4f}")
print(f"   RMSE: {rmse_lr:.4f}")
print(f"   Coeficientes: {dict(zip(selected_features, lr.coef_.round(4)))}")
print(f"   Intercepto: {lr.intercept_:.4f}")

# --- Modelo 2: Regresion Polinomial Grado 2 ---
print("\n[13] Modelo 2: Regresion Polinomial (Grado 2)")
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

lr_poly2 = LinearRegression()
lr_poly2.fit(X_train_poly2, y_train)
y_pred_poly2 = lr_poly2.predict(X_test_poly2)

r2_poly2 = r2_score(y_test, y_pred_poly2)
mae_poly2 = mean_absolute_error(y_test, y_pred_poly2)
rmse_poly2 = np.sqrt(mean_squared_error(y_test, y_pred_poly2))

results['Polinomial G2'] = {'R2': r2_poly2, 'MAE': mae_poly2, 'RMSE': rmse_poly2}

print(f"   R²:  {r2_poly2:.4f}")
print(f"   MAE: {mae_poly2:.4f}")
print(f"   RMSE: {rmse_poly2:.4f}")
print(f"   Features generadas: {X_train_poly2.shape[1]}")

# --- Modelo 3: Regresion Polinomial Grado 3 ---
print("\n[14] Modelo 3: Regresion Polinomial (Grado 3)")
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

lr_poly3 = LinearRegression()
lr_poly3.fit(X_train_poly3, y_train)
y_pred_poly3 = lr_poly3.predict(X_test_poly3)

r2_poly3 = r2_score(y_test, y_pred_poly3)
mae_poly3 = mean_absolute_error(y_test, y_pred_poly3)
rmse_poly3 = np.sqrt(mean_squared_error(y_test, y_pred_poly3))

results['Polinomial G3'] = {'R2': r2_poly3, 'MAE': mae_poly3, 'RMSE': rmse_poly3}

print(f"   R²:  {r2_poly3:.4f}")
print(f"   MAE: {mae_poly3:.4f}")
print(f"   RMSE: {rmse_poly3:.4f}")
print(f"   Features generadas: {X_train_poly3.shape[1]}")

# --- Comparativa de modelos ---
print("\n" + "=" * 70)
print("COMPARATIVA DE MODELOS")
print("=" * 70)
df_results = pd.DataFrame(results).T
df_results = df_results.sort_values('R2', ascending=False)
print(df_results.round(4).to_string())

# --- Seleccion del mejor modelo ---
best_model_name = df_results.index[0]
print(f"\n   Mejor modelo: {best_model_name}")

# --- Guardar el mejor modelo y el escalador ---
print("\n[15] Guardando modelo y escalador...")
if best_model_name == 'Lineal Multiple':
    best_model = lr
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'features': selected_features,
        'degree': 1,
        'model_name': best_model_name,
        'metrics': results[best_model_name]
    }, 'caso2_wine/modelo_wine.pkl')
elif best_model_name == 'Polinomial G2':
    best_model = lr_poly2
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'poly': poly2,
        'features': selected_features,
        'degree': 2,
        'model_name': best_model_name,
        'metrics': results[best_model_name]
    }, 'caso2_wine/modelo_wine.pkl')
else:
    best_model = lr_poly3
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'poly': poly3,
        'features': selected_features,
        'degree': 3,
        'model_name': best_model_name,
        'metrics': results[best_model_name]
    }, 'caso2_wine/modelo_wine.pkl')

print("   Guardado: caso2_wine/modelo_wine.pkl")

# --- Graficos ---
print("\n[16] Generando graficos...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Predicciones vs Real — Lineal
axes[0].scatter(y_test, y_pred_lr, alpha=0.5, color='steelblue', edgecolors='k', linewidth=0.5)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Valor Real')
axes[0].set_ylabel('Prediccion')
axes[0].set_title(f'Lineal Multiple (R²={r2_lr:.3f})')
axes[0].grid(True, alpha=0.3)

# Predicciones vs Real — Polinomial G2
axes[1].scatter(y_test, y_pred_poly2, alpha=0.5, color='coral', edgecolors='k', linewidth=0.5)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1].set_xlabel('Valor Real')
axes[1].set_ylabel('Prediccion')
axes[1].set_title(f'Polinomial G2 (R²={r2_poly2:.3f})')
axes[1].grid(True, alpha=0.3)

# Predicciones vs Real — Polinomial G3
axes[2].scatter(y_test, y_pred_poly3, alpha=0.5, color='mediumseagreen', edgecolors='k', linewidth=0.5)
axes[2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[2].set_xlabel('Valor Real')
axes[2].set_ylabel('Prediccion')
axes[2].set_title(f'Polinomial G3 (R²={r2_poly3:.3f})')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Predicciones vs Valores Reales — Comparativa de Modelos', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('caso2_wine/comparativa_modelos.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardado: caso2_wine/comparativa_modelos.png")

# Grafico de barras comparativo
fig, ax = plt.subplots(figsize=(10, 6))
models = list(results.keys())
r2_values = [results[m]['R2'] for m in models]
colors = ['steelblue', 'coral', 'mediumseagreen']
bars = ax.bar(models, r2_values, color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_title('Comparativa R² — Modelos de Regresion')
ax.set_ylim(0, max(r2_values) * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('caso2_wine/comparativa_r2.png', dpi=150)
plt.close()
print("   Guardado: caso2_wine/comparativa_r2.png")

print("\n" + "=" * 70)
print("FASE A + B COMPLETADA")
print("=" * 70)
print(f"Mejor modelo: {best_model_name}")
print(f"R²: {results[best_model_name]['R2']:.4f}")
print(f"MAE: {results[best_model_name]['MAE']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.4f}")
print("\nArchivos generados:")
print("  - caso2_wine/modelo_wine.pkl")
print("  - caso2_wine/matriz_correlacion.png")
print("  - caso2_wine/comparativa_modelos.png")
print("  - caso2_wine/comparativa_r2.png")
