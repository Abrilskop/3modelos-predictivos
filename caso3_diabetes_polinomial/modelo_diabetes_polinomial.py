import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

FEATURE_LABELS = {
    "age": "Edad (años)",
    "sex": "Sexo (1 = Femenino, 2 = Masculino)",
    "bmi": "Índice de Masa Corporal (IMC)",
    "bp": "Presión arterial media",
    "s1": "Colesterol total (tc)",
    "s2": "Lipoproteínas de baja densidad (ldl)",
    "s3": "Lipoproteínas de alta densidad (hdl)",
    "s4": "Colesterol total / HDL (tch)",
    "s5": "Logaritmo de triglicéridos séricos (ltg)",
    "s6": "Nivel de glucosa en sangre (glu)",
}


def entrenar_modelo():
    diabetes_data = load_diabetes(scaled=False)
    df_diabetes = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    df_diabetes["target"] = diabetes_data.target

    X = df_diabetes.drop(columns=["target"])
    y = df_diabetes["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Creamos un pipeline que genera características polinomiales de grado 2 y luego aplica regresión lineal
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear", LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Asegurar que el atributo feature_names_in_ esté presente
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = X.columns.tolist()
        
    y_pred = model.predict(X_test)

    metricas = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, X, metricas


def obtener_rangos(X):
    rangos = {}
    for col in X.columns:
        rangos[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "default": float(X[col].median()),
        }
    return rangos


def predecir_progresion(model, valores):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    input_data = pd.DataFrame([valores], columns=feature_names)
    return float(model.predict(input_data)[0])


def predecir_progresion_consola():
    model, X, metricas = entrenar_modelo()

    poly_step = model.named_steps["poly"]
    linear_step = model.named_steps["linear"]
    poly_features = poly_step.get_feature_names_out(X.columns)

    print("--- Coeficientes del Modelo Polinomial (Grado 2) ---")
    for feat, coef in zip(poly_features, linear_step.coef_):
        # Solo imprimimos coeficientes significativos para no saturar la consola
        if abs(coef) > 0.0001:
            print(f"{feat}: {coef:.4f}")
    print(f"Intercepto: {linear_step.intercept_:.4f}\n")

    print("--- Metricas de Evaluacion ---")
    print(f"Error Cuadratico Medio (MSE): {metricas['mse']:.4f}")
    print(f"Coeficiente de Determinacion (R2): {metricas['r2']:.4f}\n")

    print("==================================================")
    print("  PREDICCION POLINOMIAL DE LA DIABETES")
    print("==================================================")
    print("Ingrese los siguientes parametros clinicos:")

    try:
        valores = []
        for col in X.columns:
            etiqueta = FEATURE_LABELS.get(col, col)
            valor = float(input(f"{etiqueta}: "))
            valores.append(valor)

        prediction = predecir_progresion(model, valores)

        print("\n--------------------------------------------------")
        print(f"Progreso cuantitativo de la enfermedad predicho: {prediction:.2f}")
        print("--------------------------------------------------")
    except ValueError:
        print("\n[Error] Por favor, ingrese valores numericos validos.")


if __name__ == "__main__":
    predecir_progresion_consola()
