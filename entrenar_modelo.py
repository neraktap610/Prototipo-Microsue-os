"""
Entrenamiento del Modelo de Machine Learning - Detección de Fatiga
------------------------------------------------------------------------
Lee el dataset generado por 'recolectar_datos.py' (dataset_fatiga.csv)
y entrena un clasificador Random Forest para distinguir entre los
estados: normal, fatiga y microsueño, usando como características el
EAR, el MAR y la duración del cierre ocular.

Esto reemplaza el enfoque de "umbral fijo" por un modelo de IA real y
entrenado con datos, cumpliendo formalmente el objetivo específico de
"desarrollar el modelo de inteligencia artificial".

INSTALACIÓN (una sola vez):
    python -m pip install pandas scikit-learn joblib matplotlib seaborn

EJECUCIÓN:
    python entrenar_modelo.py
"""

import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

CSV_PATH = "dataset_fatiga.csv"
MODELO_PATH = "modelo_fatiga.pkl"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No se encontró '{CSV_PATH}'.")
        print("Corre primero 'recolectar_datos.py' para generar el dataset.")
        return

    datos = pd.read_csv(CSV_PATH)
    print(f"Dataset cargado: {len(datos)} muestras en total.\n")
    print("Distribución de clases:")
    print(datos["etiqueta"].value_counts())
    print()

    if len(datos) < 30:
        print("⚠️  Aviso: el dataset es muy pequeño todavía (menos de 30 muestras).")
        print("El modelo puede no aprender bien. Se recomienda recolectar más datos")
        print("(idealmente 150-200 muestras por clase, entre todo el grupo) antes")
        print("de confiar en los resultados de la validación final.\n")

    X = datos[["ear", "mar", "duracion_cierre"]]
    y = datos["etiqueta"]

    # División en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}\n")

    # Entrenamiento del modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",  # ayuda si hay clases desbalanceadas
    )
    modelo.fit(X_train, y_train)

    # Evaluación
    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)

    print("=" * 60)
    print("RESULTADOS DE VALIDACIÓN DEL MODELO")
    print("=" * 60)
    print(f"Exactitud (accuracy): {exactitud:.2%}\n")
    print("Reporte de clasificación (precisión, sensibilidad, F1 por clase):")
    print(classification_report(y_test, predicciones, zero_division=0))

    # Matriz de confusión (se guarda como imagen para el documento)
    etiquetas_unicas = sorted(y.unique())
    matriz = confusion_matrix(y_test, predicciones, labels=etiquetas_unicas)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=etiquetas_unicas)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de Confusión - Modelo de Detección de Fatiga")
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    print("\nMatriz de confusión guardada como 'matriz_confusion.png'")
    print("(puedes insertar esta imagen directamente en el documento del proyecto)")

    # Importancia de cada característica (útil para explicar el modelo en la sustentación)
    importancias = modelo.feature_importances_
    print("\nImportancia de cada característica en la decisión del modelo:")
    for nombre, importancia in zip(X.columns, importancias):
        print(f"  - {nombre}: {importancia:.2%}")

    # Guardar el modelo entrenado
    joblib.dump(modelo, MODELO_PATH)
    print(f"\n✅ Modelo entrenado y guardado como '{MODELO_PATH}'")

    # Guardar las métricas en un archivo JSON, para que el dashboard las lea
    reporte_dict = classification_report(y_test, predicciones, zero_division=0, output_dict=True)
    metricas = {
        "exactitud": exactitud,
        "total_muestras": len(datos),
        "muestras_entrenamiento": len(X_train),
        "muestras_prueba": len(X_test),
        "reporte_por_clase": reporte_dict,
        "importancia_caracteristicas": {
            nombre: float(importancia) for nombre, importancia in zip(X.columns, importancias)
        },
        "hiperparametros": {
            "algoritmo": "Random Forest",
            "n_estimators": modelo.n_estimators,
            "max_depth": modelo.max_depth,
            "class_weight": "balanced",
            "test_size": 0.2,
        },
        "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("Métricas guardadas en 'metricas_modelo.json'")
    print("Ahora puedes usar la versión con IA de 'deteccion_fatiga.py', o abrir el dashboard con 'streamlit run dashboard_fatiga.py'")


if __name__ == "__main__":
    main()