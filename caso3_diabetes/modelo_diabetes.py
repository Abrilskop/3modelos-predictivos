import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

    model = LinearRegression()
    model.fit(X_train, y_train)
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
    input_data = pd.DataFrame([valores], columns=model.feature_names_in_)
    return float(model.predict(input_data)[0])


def predecir_progresion_consola():
    model, X, metricas = entrenar_modelo()

    print("--- Coeficientes del Modelo (Datos No Escalados) ---")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"{feat}: {coef:.4f}")
    print(f"Intercepto: {model.intercept_:.4f}\n")

    print("--- Metricas de Evaluacion ---")
    print(f"Error Cuadratico Medio (MSE): {metricas['mse']:.4f}")
    print(f"Coeficiente de Determinacion (R2): {metricas['r2']:.4f}\n")

    print("==================================================")
    print("  PREDICCION DE LA PROGRESION DE LA DIABETES")
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
