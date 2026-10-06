"""
Servidor FastAPI para el análisis biomecánico EN VIVO por webcam.

Usa tus funciones reales de biomecanica_utils.py -- no es una simulación.
Por ahora solo calcula postura del tronco y de la cabeza (son cálculos
por frame). Cadencia y fatiga necesitan una ventana de varios frames o
un modelo entrenado, y quedan pendientes para una siguiente vuelta.

Instalar dependencias (una sola vez):
    pip install fastapi "uvicorn[standard]" opencv-python numpy

Correr:
    uvicorn server:app --reload

Después abrí http://localhost:8000 en el navegador y dale a "Iniciar cámara".
"""

import base64
import os
import shutil
import tempfile
import time
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from biomecanica_utils import (
    crear_tracker,
    seleccionar_persona_principal,
    procesar_video,
    interpolar_nans,
    suavizar_keypoints,
    suavizar_serie,
    calcular_postura,
    calcular_inclinacion_cabeza,
    detectar_desviaciones_adaptativas,
    calcular_metricas_paso,
    construir_buffer,
    perfil_fatiga_sesion,
    TransformerAutoencoder,
    clasificar_pisada,
    resumen_patron_pisada,
    exportar_clip_alerta,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_BIG_TOE,
    RIGHT_BIG_TOE,
    formatear_tiempo,
)

os.makedirs("clips", exist_ok=True)

# Consejos genéricos por tipo de alerta -- NO es asesoramiento médico ni de
# un entrenador certificado, son pautas generales de técnica de carrera.
RECOMENDACIONES = {
    ("Tronco", "adelante"): "Revisá no inclinarte demasiado hacia adelante -- puede sobrecargar la zona lumbar. Probá mantener el tronco más erguido, con la mirada al frente.",
    ("Tronco", "atrás"): "Inclinarte hacia atrás suele pasar al frenar con el talón muy adelante del cuerpo. Probá que el pie aterrice más cerca de la vertical de tu cadera.",
    ("Cabeza", "adelante"): "Cabeza proyectada hacia adelante (mirar el piso) tensiona el cuello. Probá llevar la mirada al horizonte, no a tus pies.",
    ("Cabeza", "atrás"): "Cabeza muy hacia atrás puede ser señal de fatiga o de compensar el tronco. Si se repite seguido, considerá bajar el ritmo.",
}

RUTA_MODELO_FATIGA = "modelo_fatiga_general.pt"
MODELO_FATIGA = None
if os.path.exists(RUTA_MODELO_FATIGA):
    MODELO_FATIGA = TransformerAutoencoder()
    MODELO_FATIGA.load_state_dict(torch.load(RUTA_MODELO_FATIGA, map_location="cpu"))
    MODELO_FATIGA.eval()
    print(f"Modelo de fatiga cargado desde {RUTA_MODELO_FATIGA}")
else:
    print(f"AVISO: no se encontró {RUTA_MODELO_FATIGA} -- la barra de fatiga quedará deshabilitada")

app = FastAPI()
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

# Un solo tracker, compartido -- funciona igual que dentro de procesar_video,
# solo que acá el "video" llega frame a frame por WebSocket en vez de un
# archivo. "lightweight" + sin tracking es lo que dio ~34 ms/frame en tus
# propias mediciones (ver conversación).
TRACKER = crear_tracker(mode="lightweight", backend="onnxruntime", device="cpu", tracking=False)

CONFIANZA_MIN = 0.3


class EstadoSesion:
    """
    Línea base y suavizado, calculados en vivo (streaming), a diferencia
    de calcular_linea_base_postura / detectar_desviaciones_posturales,
    que trabajan sobre un array completo ya grabado.

    Simplificación a propósito: la línea base son los primeros
    SEGUNDOS_BASELINE segundos de ESTA conexión (no hay forma de "saltar
    al segundo 15" como en el análisis por lotes, porque en vivo no existe
    el resto del video todavía).
    """

    SEGUNDOS_BASELINE = 8
    UMBRAL_Z = 2.0
    VENTANA_SUAVIZADO = 5  # frames, promedio móvil simple (no OneEuroFilter)

    def __init__(self):
        self.inicio = time.time()
        self.muestras_base = {"tronco": [], "cabeza": []}
        self.media = {"tronco": None, "cabeza": None}
        self.std = {"tronco": None, "cabeza": None}
        self.historial = {"tronco": [], "cabeza": []}

    def _suavizar(self, nombre, valor):
        h = self.historial[nombre]
        h.append(valor)
        if len(h) > self.VENTANA_SUAVIZADO:
            h.pop(0)
        return float(np.mean(h))

    def procesar(self, keypoints_persona):
        # calcular_postura/calcular_inclinacion_cabeza esperan (N, 17, 2);
        # acá tenemos un solo frame, así que agregamos esa dimensión con [None].
        tronco = float(calcular_postura(keypoints_persona[None, ...])[0])
        cabeza = float(calcular_inclinacion_cabeza(keypoints_persona[None, ...])[0])

        tronco = self._suavizar("tronco", tronco)
        cabeza = self._suavizar("cabeza", cabeza)

        transcurrido = time.time() - self.inicio

        if transcurrido < self.SEGUNDOS_BASELINE:
            self.muestras_base["tronco"].append(tronco)
            self.muestras_base["cabeza"].append(cabeza)
            return {
                "estado": "calibrando",
                "segundos_restantes": round(self.SEGUNDOS_BASELINE - transcurrido, 1),
                "tronco": round(tronco, 1),
                "cabeza": round(cabeza, 1),
            }

        if self.media["tronco"] is None:
            for nombre in ("tronco", "cabeza"):
                muestras = self.muestras_base[nombre]
                self.media[nombre] = float(np.mean(muestras))
                self.std[nombre] = float(np.std(muestras)) + 1e-6

        z = {
            "tronco": (tronco - self.media["tronco"]) / self.std["tronco"],
            "cabeza": (cabeza - self.media["cabeza"]) / self.std["cabeza"],
        }

        alerta = None
        for nombre, etiqueta in (("tronco", "Tronco"), ("cabeza", "Cabeza")):
            if abs(z[nombre]) > self.UMBRAL_Z:
                direccion = "adelante" if z[nombre] > 0 else "atrás"
                alerta = f"{etiqueta} inclinado hacia {direccion} más de lo normal para vos (z={z[nombre]:.1f})"
                break  # una alerta a la vez, la primera que cruce el umbral

        return {
            "estado": "listo",
            "tronco": round(tronco, 1),
            "cabeza": round(cabeza, 1),
            "base_tronco": round(self.media["tronco"], 1),
            "base_cabeza": round(self.media["cabeza"], 1),
            "z_tronco": round(z["tronco"], 2),
            "z_cabeza": round(z["cabeza"], 2),
            "alerta": alerta,
        }


@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend.html", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    estado = EstadoSesion()
    try:
        while True:
            data_url = await websocket.receive_text()
            # el navegador manda algo tipo "data:image/jpeg;base64,XXXXX"
            _, b64data = data_url.split(",", 1)
            jpg_bytes = base64.b64decode(b64data)
            frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            keypoints, scores = TRACKER(frame)
            persona, confianza = seleccionar_persona_principal(keypoints, scores)
            detectado = persona is not None and (confianza is None or confianza >= CONFIANZA_MIN)

            if not detectado:
                await websocket.send_json({"estado": "sin_persona"})
                continue

            resultado = estado.procesar(persona)
            resultado["keypoints"] = persona.tolist()  # para dibujar el esqueleto en el navegador
            await websocket.send_json(resultado)
    except WebSocketDisconnect:
        pass


def _mensaje_alerta(alerta, parte):
    """Igual que generar_mensaje_alerta, pero sirve tanto para tronco como
    para cabeza (la de biomecanica_utils solo dice "Tronco")."""
    inicio = formatear_tiempo(alerta["inicio_seg"])
    fin = formatear_tiempo(alerta["fin_seg"])
    if alerta.get("posible_incidente"):
        return (f"[{inicio}-{fin}] "
                f"Posible caída o incidente (z={alerta['z_promedio']:.1f}) -- "
                f"no se interpreta como desviación de técnica")
    delta = alerta.get("delta_grados")
    detalle_grados = f", Δ={delta:+.1f}°" if delta is not None else ""
    return (f"[{inicio}-{fin}] "
            f"{parte} inclinado hacia {alerta['direccion']} más de lo normal para vos "
            f"(z={alerta['z_promedio']:.1f}{detalle_grados})")


@app.post("/upload")
async def upload_video(archivo: UploadFile = File(...)):
    """
    Análisis por lotes de un video ya grabado: la misma idea que tu
    notebook 05, resumida a un endpoint. Calcula postura, cabeza,
    cadencia, impacto, asimetría, patrón de pisada y (si está el modelo)
    fatiga -- y para las alertas más severas, exporta un clip corto con
    el esqueleto dibujado, más una recomendación genérica según el tipo
    de desviación.
    """
    suffix = os.path.splitext(archivo.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(archivo.file, tmp)
        ruta_tmp = tmp.name

    tracker_local = crear_tracker(mode="lightweight", backend="onnxruntime",
                                   device="cpu", tracking=False, con_pies=True)
    try:
        keypoints, total_frames, _, fps = procesar_video(ruta_tmp, tracker_local, n_keypoints=26)

        frames_validos = ~np.isnan(keypoints).any(axis=(1, 2))
        keypoints_interp = interpolar_nans(keypoints)
        keypoints_suavizados = suavizar_keypoints(keypoints_interp, fps=fps)

        metricas_paso = calcular_metricas_paso(keypoints_suavizados, fps=fps)

        postura = calcular_postura(keypoints_suavizados)
        cabeza = calcular_inclinacion_cabeza(keypoints_suavizados)

        # Misma idea que ya armamos en el notebook: suavizar según la
        # cadencia real de ESTE video antes de calcular línea base y z-score.
        cadencia_valida = metricas_paso["cadencia"] if metricas_paso["cadencia"] else 140.0
        frames_por_ciclo = max(int((60 / cadencia_valida) * fps), 1)
        ventana_suavizado = frames_por_ciclo * 2
        postura_suave = suavizar_serie(postura, ventana=ventana_suavizado)
        cabeza_suave = suavizar_serie(cabeza, ventana=ventana_suavizado)

        duracion_seg = total_frames / fps
        duracion_baseline = min(10.0, max(duracion_seg / 3, 1.0))
        inicio_baseline = min(15.0, max(duracion_seg / 6, 0.0))

        # Tronco y cabeza: las dos derivan naturalmente a lo largo de la
        # sesión (ver conversación), así que las dos usan la versión
        # adaptativa -- compara contra la tendencia reciente, no contra un
        # único punto de referencia al inicio.
        alertas_postura = detectar_desviaciones_adaptativas(
            postura_suave, ventana_seg=75, umbral_z=2.5, duracion_sostenida_seg=1.0,
            fps=fps, frames_validos=frames_validos, umbral_grados_min=5.0,
        )
        alertas_cabeza = detectar_desviaciones_adaptativas(
            cabeza_suave, ventana_seg=75, umbral_z=2.5, duracion_sostenida_seg=1.0,
            fps=fps, frames_validos=frames_validos, umbral_grados_min=8.0,
        )

        alertas_todas = (
            [dict(a, parte="Tronco") for a in alertas_postura]
            + [dict(a, parte="Cabeza") for a in alertas_cabeza]
        )
        alertas_todas.sort(key=lambda a: a["inicio_seg"])

        # Clips + recomendación solo para las N alertas más severas (en
        # grados reales, no z-score) -- exportar un clip por cada alerta
        # sería lento en una sesión con muchas, y la mayoría no aporta
        # nada nuevo para revisar a mano.
        MAX_CLIPS = 4
        indices_severos = sorted(
            range(len(alertas_todas)),
            key=lambda i: abs(alertas_todas[i]["delta_grados"]),
            reverse=True,
        )[:MAX_CLIPS]

        alertas = []
        for i, a in enumerate(alertas_todas):
            item = {
                "segundo": a["inicio_seg"],
                "texto": _mensaje_alerta(a, a["parte"]),
                "recomendacion": RECOMENDACIONES.get((a["parte"], a["direccion"])),
                "clip_url": None,
            }
            if i in indices_severos:
                try:
                    nombre_clip = f"{uuid.uuid4().hex}.mp4"
                    ruta_clip = os.path.join("clips", nombre_clip)
                    exportar_clip_alerta(
                        ruta_tmp, tracker_local,
                        int(a["inicio_seg"] * fps), int(a["fin_seg"] * fps),
                        ruta_clip, fps=fps, margen_seg=1.0,
                    )
                    item["clip_url"] = f"/clips/{nombre_clip}"
                except Exception as err:
                    print(f"No se pudo exportar el clip de la alerta en {a['inicio_seg']:.1f}s: {err}")
            alertas.append(item)

        # Patrón de pisada (necesita los 9 puntos extra de pies, por eso
        # este endpoint usa con_pies=True a diferencia del resto del proyecto)
        resultados_izq = clasificar_pisada(
            keypoints_suavizados, metricas_paso["contactos_izq"], LEFT_HEEL, LEFT_BIG_TOE)
        resultados_der = clasificar_pisada(
            keypoints_suavizados, metricas_paso["contactos_der"], RIGHT_HEEL, RIGHT_BIG_TOE)
        patron_pisada = resumen_patron_pisada(resultados_izq, resultados_der)

        fatiga = {"disponible": False}
        if MODELO_FATIGA is not None:
            # El autoencoder se entrenó con 17 puntos (34 valores por frame,
            # ver notebook 04) -- acá keypoints_suavizados tiene 26 (con
            # pies, para la pisada), así que hay que recortar a los
            # primeros 17 antes de armar el buffer. Son los mismos puntos
            # COCO-17 de siempre, en el mismo orden.
            buffer_sesion = construir_buffer(keypoints_suavizados[:, :17, :], window_size=30, stride=1)
            tiempos_f, error_f = perfil_fatiga_sesion(MODELO_FATIGA, buffer_sesion, fps=fps)

            en_baseline = (tiempos_f >= inicio_baseline) & (tiempos_f < inicio_baseline + duracion_baseline)
            base_error = (float(np.median(error_f[en_baseline])) if en_baseline.any()
                          else float(np.median(error_f[:max(1, len(error_f) // 10)])))

            tramo_final = error_f[int(len(error_f) * 2 / 3):]
            ratio_final = float(np.median(tramo_final) / base_error) if base_error > 0 else 1.0
            barra_pct = float(np.clip((ratio_final - 1.0), 0.0, 1.0) * 100)

            fatiga = {
                "disponible": True,
                "ratio_final_vs_base": round(ratio_final, 2),
                "barra_pct": round(barra_pct, 0),
            }

        return {
            "duracion_seg": round(duracion_seg, 1),
            "fps": round(fps, 1),
            "cadencia": round(metricas_paso["cadencia"], 1),
            "impacto": round(metricas_paso["impacto"], 2),
            "asimetria": (round(metricas_paso["asimetria"], 1)
                          if metricas_paso["asimetria"] is not None else None),
            "promedio_tronco": round(float(np.mean(postura_suave)), 1),
            "promedio_cabeza": round(float(np.mean(cabeza_suave)), 1),
            "patron_pisada": patron_pisada,
            "fatiga": fatiga,
            "alertas": alertas,
        }
    finally:
        os.remove(ruta_tmp)
