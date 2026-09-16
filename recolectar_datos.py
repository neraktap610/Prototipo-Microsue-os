"""
Recolector de Datos Etiquetados - Sistema de Detección de Fatiga
--------------------------------------------------------------------
Este script muestra tu cámara en vivo, calcula las características
faciales (EAR, MAR, duración de cierre ocular) en cada cuadro, y te
permite ETIQUETAR manualmente qué estado representa ese momento
(normal, fatiga o microsueño) con solo presionar una tecla.

Con esto se construye el dataset (dataset_fatiga.csv) que luego usará
'entrenar_modelo.py' para entrenar un modelo real de Machine Learning.

CÓMO USARLO (¡esto es lo más importante!):
    - Presiona 'n' varias veces mientras miras normal a la cámara
      (parpadeando de forma natural) -> etiqueta "normal"
    - Presiona 'f' varias veces mientras simulas fatiga (ojos entrecerrados,
      pestañeo lento y prolongado) -> etiqueta "fatiga"
    - Presiona 'm' varias veces mientras simulas un microsueño (cierra los
      ojos completamente por 2-3 segundos) -> etiqueta "microsueno"
    - Presiona 'q' para salir y guardar todo

RECOMENDACIÓN: Repite esto en varias sesiones (distintos días, distinta
luz, e incluso con los 3 integrantes del grupo turnándose frente a la
cámara) para tener un dataset más robusto y variado. Mientras más
ejemplos de cada clase, mejor aprenderá el modelo. Apunten a un mínimo
de 150-200 ejemplos por clase entre todos.

INSTALACIÓN (si falta algo):
    python -m pip install opencv-python mediapipe numpy pandas
"""

import cv2
import numpy as np
import time
import os
import csv

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------------

OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
OJO_DERECHO = [33, 160, 158, 133, 153, 144]
BOCA_VERTICAL = [13, 14]   # labio superior/inferior internos
BOCA_HORIZONTAL = [78, 308]  # comisuras de la boca

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

CSV_PATH = "dataset_fatiga.csv"
EAR_UMBRAL_REFERENCIA = 0.21  # solo para mostrar en pantalla, no decide nada aquí


def asegurar_modelo():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("Descargando modelo de detección facial...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def calcular_distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calcular_ear(puntos_ojo, landmarks, ancho, alto):
    coords = [(int(landmarks[i].x * ancho), int(landmarks[i].y * alto)) for i in puntos_ojo]
    vertical_1 = calcular_distancia(coords[1], coords[5])
    vertical_2 = calcular_distancia(coords[2], coords[4])
    horizontal = calcular_distancia(coords[0], coords[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calcular_mar(landmarks, ancho, alto):
    top = landmarks[BOCA_VERTICAL[0]]
    bottom = landmarks[BOCA_VERTICAL[1]]
    left = landmarks[BOCA_HORIZONTAL[0]]
    right = landmarks[BOCA_HORIZONTAL[1]]

    p_top = (int(top.x * ancho), int(top.y * alto))
    p_bottom = (int(bottom.x * ancho), int(bottom.y * alto))
    p_left = (int(left.x * ancho), int(left.y * alto))
    p_right = (int(right.x * ancho), int(right.y * alto))

    vertical = calcular_distancia(p_top, p_bottom)
    horizontal = calcular_distancia(p_left, p_right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def inicializar_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ear", "mar", "duracion_cierre", "etiqueta"])


def guardar_muestra(ear, mar, duracion_cierre, etiqueta):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([round(ear, 4), round(mar, 4), round(duracion_cierre, 3), etiqueta])


def main():
    inicializar_csv()
    asegurar_modelo()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo acceder a la cámara.")
        return

    tiempo_inicio_cierre = None
    contador_muestras = {"normal": 0, "fatiga": 0, "microsueno": 0}

    print("=" * 65)
    print("RECOLECCIÓN DE DATOS ETIQUETADOS")
    print("=" * 65)
    print("Presiona:")
    print("  'n' -> etiquetar el frame actual como NORMAL")
    print("  'f' -> etiquetar el frame actual como FATIGA")
    print("  'm' -> etiquetar el frame actual como MICROSUEÑO")
    print("  'q' -> salir y guardar")
    print("=" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

        ear_promedio = 0.0
        mar = 0.0
        duracion_cierre = 0.0

        if resultado.face_landmarks:
            landmarks = resultado.face_landmarks[0]

            ear_izq = calcular_ear(OJO_IZQUIERDO, landmarks, ancho, alto)
            ear_der = calcular_ear(OJO_DERECHO, landmarks, ancho, alto)
            ear_promedio = (ear_izq + ear_der) / 2.0
            mar = calcular_mar(landmarks, ancho, alto)

            if ear_promedio < EAR_UMBRAL_REFERENCIA:
                if tiempo_inicio_cierre is None:
                    tiempo_inicio_cierre = time.time()
                duracion_cierre = time.time() - tiempo_inicio_cierre
            else:
                tiempo_inicio_cierre = None
                duracion_cierre = 0.0

            cv2.putText(frame, f"EAR: {ear_promedio:.3f}  MAR: {mar:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Cierre: {duracion_cierre:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        texto_contador = (f"Normal:{contador_muestras['normal']}  "
                           f"Fatiga:{contador_muestras['fatiga']}  "
                           f"Microsueno:{contador_muestras['microsueno']}")
        cv2.putText(frame, texto_contador, (10, alto - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "n=normal f=fatiga m=microsueno q=salir", (10, alto - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Recoleccion de Datos - Trabajo de Grado", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('n') and resultado.face_landmarks:
            guardar_muestra(ear_promedio, mar, duracion_cierre, "normal")
            contador_muestras["normal"] += 1
            print(f"[Guardado] normal | EAR={ear_promedio:.3f} MAR={mar:.3f}")
        elif tecla == ord('f') and resultado.face_landmarks:
            guardar_muestra(ear_promedio, mar, duracion_cierre, "fatiga")
            contador_muestras["fatiga"] += 1
            print(f"[Guardado] fatiga | EAR={ear_promedio:.3f} MAR={mar:.3f}")
        elif tecla == ord('m') and resultado.face_landmarks:
            guardar_muestra(ear_promedio, mar, duracion_cierre, "microsueno")
            contador_muestras["microsueno"] += 1
            print(f"[Guardado] microsueno | EAR={ear_promedio:.3f} MAR={mar:.3f}")

    cap.release()
    cv2.destroyAllWindows()

    print("\nResumen de la sesión de recolección:")
    for clase, cantidad in contador_muestras.items():
        print(f"  - {clase}: {cantidad} muestras nuevas")
    print(f"\nTotal acumulado en {CSV_PATH} (incluyendo sesiones anteriores).")
    print("Corre 'entrenar_modelo.py' cuando tengan suficientes muestras de cada clase.")


if __name__ == "__main__":
    main()