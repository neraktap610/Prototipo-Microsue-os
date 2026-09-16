"""
Sistema de Detección Temprana de Fatiga y Microsueños (v4 - con Machine Learning)
------------------------------------------------------------------------------------
Versión con modelo de IA real (Random Forest) entrenado con datos propios,
en vez de un umbral fijo. Si el modelo entrenado no existe todavía
(modelo_fatiga.pkl), el sistema funciona en modo de respaldo con el
umbral EAR clásico, y te avisa que debes entrenar el modelo.

FLUJO COMPLETO DEL PROYECTO:
    1. python recolectar_datos.py   -> genera dataset_fatiga.csv
    2. python entrenar_modelo.py    -> genera modelo_fatiga.pkl
    3. python deteccion_fatiga.py   -> usa el modelo entrenado (este archivo)

INSTALACIÓN:
    python -m pip install opencv-python mediapipe numpy playsound==1.2.2 paho-mqtt==1.6.1 scikit-learn joblib

EJECUCIÓN:
    python deteccion_fatiga.py

Presiona 'q' con la ventana de video activa para salir.
"""

import cv2
import numpy as np
import time
import threading
import os
import sqlite3
import random
import urllib.request

import paho.mqtt.client as mqtt

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    from playsound import playsound
    SONIDO_DISPONIBLE = True
except ImportError:
    SONIDO_DISPONIBLE = False
    print("Aviso: 'playsound' no está instalado. Solo habrá alerta visual.")

try:
    import joblib
    JOBLIB_DISPONIBLE = True
except ImportError:
    JOBLIB_DISPONIBLE = False

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------------------------

# --- Umbral de respaldo (solo se usa si el modelo de IA no está entrenado) ---
EAR_UMBRAL = 0.21
SEGUNDOS_ALERTA_FATIGA = 0.5
SEGUNDOS_ALERTA_MICROSUENO = 1.5

OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
OJO_DERECHO = [33, 160, 158, 133, 153, 144]
BOCA_VERTICAL = [13, 14]
BOCA_HORIZONTAL = [78, 308]

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

MODELO_ML_PATH = "modelo_fatiga.pkl"

DB_PATH = "eventos_fatiga.db"

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "fatiga/conductor1/alerta"

mqtt_client = mqtt.Client(client_id=f"PythonFatiga-{random.randint(1000,9999)}")

CONDUCTOR_ID = "conductor_prueba_1"


def conectar_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"Conectado al broker MQTT ({MQTT_BROKER}), listo para enviar alertas.")
    except Exception as e:
        print(f"Aviso: no se pudo conectar al broker MQTT ({e}). "
              "La detección seguirá funcionando, pero sin avisar al ESP32.")


def enviar_alerta_microsueno():
    try:
        mqtt_client.publish(MQTT_TOPIC, "MICROSUEÑO")
        print(f"[MQTT] Alerta enviada al ESP32 en el canal '{MQTT_TOPIC}'")
    except Exception as e:
        print(f"[MQTT] Error al enviar alerta: {e}")


# --------------------------------------------------------------------------
# 2. BASE DE DATOS
# --------------------------------------------------------------------------

def inicializar_base_datos():
    conexion = sqlite3.connect(DB_PATH)
    cursor = conexion.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conductor_id TEXT NOT NULL,
            tipo_evento TEXT NOT NULL,
            valor_ear REAL,
            fecha_hora TEXT NOT NULL
        )
    """)
    conexion.commit()
    conexion.close()


def guardar_evento(conductor_id, tipo_evento, valor_ear):
    conexion = sqlite3.connect(DB_PATH)
    cursor = conexion.cursor()
    fecha_hora = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO eventos (conductor_id, tipo_evento, valor_ear, fecha_hora) "
        "VALUES (?, ?, ?, ?)",
        (conductor_id, tipo_evento, valor_ear, fecha_hora)
    )
    conexion.commit()
    conexion.close()
    return fecha_hora


def obtener_resumen_sesion(conductor_id):
    conexion = sqlite3.connect(DB_PATH)
    cursor = conexion.cursor()
    cursor.execute(
        "SELECT tipo_evento, COUNT(*) FROM eventos WHERE conductor_id = ? "
        "GROUP BY tipo_evento",
        (conductor_id,)
    )
    resultados = cursor.fetchall()
    conexion.close()
    return resultados


# --------------------------------------------------------------------------
# 3. MODELO DE MACHINE LEARNING
# --------------------------------------------------------------------------

def cargar_modelo_ml():
    if JOBLIB_DISPONIBLE and os.path.exists(MODELO_ML_PATH):
        modelo = joblib.load(MODELO_ML_PATH)
        print(f"✅ Modelo de IA cargado ({MODELO_ML_PATH}). Usando clasificación por Machine Learning.")
        return modelo
    else:
        print("⚠️  No se encontró un modelo entrenado (modelo_fatiga.pkl).")
        print("El sistema funcionará con el umbral de respaldo mientras tanto.")
        print("Corre 'recolectar_datos.py' y luego 'entrenar_modelo.py' para activar el modelo de IA.\n")
        return None


# --------------------------------------------------------------------------
# 4. DESCARGA AUTOMÁTICA DEL MODELO DE MEDIAPIPE
# --------------------------------------------------------------------------

def asegurar_modelo_mediapipe():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo de detección facial (una sola vez)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado correctamente.")


# --------------------------------------------------------------------------
# 5. FUNCIONES DE CÁLCULO DE CARACTERÍSTICAS
# --------------------------------------------------------------------------

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


def reproducir_alerta():
    if SONIDO_DISPONIBLE:
        try:
            playsound("alerta.mp3")
        except Exception:
            print("\a")
    else:
        print("\a")


# --------------------------------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# --------------------------------------------------------------------------

def main():
    inicializar_base_datos()
    asegurar_modelo_mediapipe()
    conectar_mqtt()
    modelo_ml = cargar_modelo_ml()

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
    alerta_fatiga_activa = False
    alerta_microsueno_activa = False

    print("Sistema iniciado. Presiona 'q' para salir.")

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

        estado_texto = "Sin rostro detectado"
        color_texto = (0, 0, 255)
        clasificacion = "normal"

        if resultado.face_landmarks:
            landmarks = resultado.face_landmarks[0]

            ear_izq = calcular_ear(OJO_IZQUIERDO, landmarks, ancho, alto)
            ear_der = calcular_ear(OJO_DERECHO, landmarks, ancho, alto)
            ear_promedio = (ear_izq + ear_der) / 2.0
            mar = calcular_mar(landmarks, ancho, alto)

            if ear_promedio < EAR_UMBRAL:
                if tiempo_inicio_cierre is None:
                    tiempo_inicio_cierre = time.time()
                duracion_cierre = time.time() - tiempo_inicio_cierre
            else:
                tiempo_inicio_cierre = None
                duracion_cierre = 0.0

            # --------------------------------------------------------
            # CLASIFICACIÓN: por Machine Learning si hay modelo,
            # si no, por el umbral de respaldo.
            # --------------------------------------------------------
            if modelo_ml is not None:
                caracteristicas = [[ear_promedio, mar, duracion_cierre]]
                clasificacion = modelo_ml.predict(caracteristicas)[0]
            else:
                if duracion_cierre >= SEGUNDOS_ALERTA_MICROSUENO:
                    clasificacion = "microsueno"
                elif duracion_cierre >= SEGUNDOS_ALERTA_FATIGA:
                    clasificacion = "fatiga"
                else:
                    clasificacion = "normal"

            # --------------------------------------------------------
            # Reacciones según la clasificación obtenida
            # --------------------------------------------------------
            if clasificacion == "microsueno":
                estado_texto = "MICROSUEÑO DETECTADO"
                color_texto = (0, 0, 255)
                if not alerta_microsueno_activa:
                    alerta_microsueno_activa = True
                    alerta_fatiga_activa = False
                    threading.Thread(target=reproducir_alerta, daemon=True).start()
                    threading.Thread(target=enviar_alerta_microsueno, daemon=True).start()
                    hora = guardar_evento(CONDUCTOR_ID, "microsueño", ear_promedio)
                    print(f"[ALERTA] Microsueño guardado en la base de datos a las {hora}")

            elif clasificacion == "fatiga":
                estado_texto = "Fatiga incipiente..."
                color_texto = (0, 165, 255)
                if not alerta_fatiga_activa:
                    alerta_fatiga_activa = True
                    alerta_microsueno_activa = False
                    hora = guardar_evento(CONDUCTOR_ID, "fatiga_incipiente", ear_promedio)
                    print(f"[INFO] Fatiga incipiente guardada a las {hora}")

            else:
                estado_texto = "Estado normal"
                color_texto = (0, 255, 0)
                alerta_fatiga_activa = False
                alerta_microsueno_activa = False

            fuente_clasificacion = "IA (Random Forest)" if modelo_ml is not None else "Umbral (respaldo)"
            cv2.putText(frame, f"EAR: {ear_promedio:.3f}  MAR: {mar:.3f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Clasificador: {fuente_clasificacion}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

        cv2.putText(frame, estado_texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2)

        cv2.imshow("Deteccion de Fatiga - Prototipo Trabajo de Grado", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nResumen de la sesión (guardado en la base de datos):")
    resumen = obtener_resumen_sesion(CONDUCTOR_ID)
    if resumen:
        for tipo, cantidad in resumen:
            print(f"  - {tipo}: {cantidad} evento(s)")
    else:
        print("  No se registraron eventos en esta sesión.")


if __name__ == "__main__":
    main()