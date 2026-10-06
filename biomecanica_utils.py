"""
biomecanica_utils.py
---------------------
Módulo compartido con todas las funciones y clases reutilizables del
proyecto de análisis biomecánico de carrera. Los 4 notebooks
(01 a 04) importan de acá -- así el código vive en un solo lugar,
y cada notebook queda corto y enfocado en su propia parte de la historia.

Para usarlo, poné este archivo en la misma carpeta que los notebooks,
y en cada uno: from biomecanica_utils import <lo que necesites>
"""

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rtmlib import Body, BodyWithFeet, PoseTracker, draw_skeleton


# ============================================================
# Índices COCO-17 (formato que entrega RTMPose vía rtmlib)
# ============================================================
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Índices adicionales de Halpe26 (BodyWithFeet) -- los primeros 17 son
# IGUALES a COCO-17 (arriba); estos son los 9 extra que agrega el modelo.
LEFT_BIG_TOE, RIGHT_BIG_TOE = 20, 21
LEFT_SMALL_TOE, RIGHT_SMALL_TOE = 22, 23
LEFT_HEEL, RIGHT_HEEL = 24, 25


# ============================================================
# 1. Extracción: RTMPose + tracking + selección de persona principal
# ============================================================
def crear_tracker(mode="balanced", backend="onnxruntime", device="cpu", con_pies=False, tracking=True):
    """
    Crea el PoseTracker (RTMPose + tracking por IoU). El tracker asigna
    un ID persistente a cada persona entre frames -- así "persona 0" en
    el frame 50 sigue siendo la misma persona en el frame 51, y no
    "quien sea que detectó primero" (importante si hay más de una
    persona en el cuadro).

    con_pies=True usa BodyWithFeet (26 puntos, formato Halpe26: los
    primeros 17 son iguales a COCO-17, más 9 puntos extra incluyendo
    talón y punta del pie) -- necesario para clasificar patrón de
    pisada. Por defecto usa Body (17 puntos, COCO-17), igual que antes.

    tracking=False desactiva el seguimiento de identidad entre frames
    (más rápido; ver mediciones de ms/frame). Con una sola persona en
    cuadro no hace falta el ID persistente, así que suele ser seguro
    desactivarlo para ganar velocidad.
    """
    modelo = BodyWithFeet if con_pies else Body
    return PoseTracker(
        modelo,
        det_frequency=5,     # re-detecta cada 5 frames, trackea en los intermedios (más rápido)
        tracking=tracking,
        to_openpose=False,   # False = mantiene formato COCO/Halpe de puntos
        mode=mode,
        backend=backend,
        device=device,
    )


def area_de_persona(kpts):
    """Área aproximada del bounding box de una persona a partir de sus keypoints."""
    x, y = kpts[:, 0], kpts[:, 1]
    return (x.max() - x.min()) * (y.max() - y.min())


def seleccionar_persona_principal(keypoints, scores=None):
    """
    De todas las personas trackeadas en un frame, elige la de bounding
    box más grande -- asumimos que es el sujeto principal (quien graba
    el video de cerca), no alguien de fondo.

    Devuelve (keypoints_persona, confianza_promedio). confianza_promedio
    es None si no se pasaron scores (no se puede evaluar confiabilidad).
    """
    if len(keypoints) == 0:
        return None, None
    areas = [area_de_persona(kpts) for kpts in keypoints]
    idx = int(np.argmax(areas))
    confianza = float(np.mean(scores[idx])) if scores is not None else None
    return keypoints[idx], confianza


def procesar_video(fuente, tracker, n_keypoints=17, confianza_min=0.3):
    """
    Procesa un video con el tracker dado.
    n_keypoints: 17 para Body (COCO-17, default), 26 para BodyWithFeet (Halpe26).

    confianza_min: confianza promedio mínima (score de RTMPose, 0-1)
    para aceptar una detección como persona real. Una detección con
    bounding box grande pero confianza muy baja probablemente NO es la
    persona real (ruido, sombra, objeto de fondo mal clasificado) --
    esos frames se tratan como "sin detección" (igual que si no hubiera
    nadie), en vez de aceptar coordenadas poco confiables.

    Devuelve (keypoints, total_frames, ultimo_frame_anotado, fps).
    """
    tracker.reset()  # por si el tracker ya se usó en otro video antes

    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir la fuente de video: {fuente}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    all_keypoints = []
    ultimo_frame_anotado = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = tracker(frame)
        persona, confianza = seleccionar_persona_principal(keypoints, scores)
        detectado = persona is not None and (confianza is None or confianza >= confianza_min)

        if detectado:
            all_keypoints.append(persona)
            ultimo_frame_anotado = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
        else:
            all_keypoints.append(np.full((n_keypoints, 2), np.nan))

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Procesados {frame_count} frames...")

    cap.release()
    return np.array(all_keypoints), frame_count, ultimo_frame_anotado, fps


def exportar_frames_segmento(fuente, tracker, frame_inicio, frame_fin, carpeta="frames_conteo"):
    """
    Guarda como imágenes (con el esqueleto dibujado) los frames de un
    segmento -- para contar contactos a mano, frame por frame, y
    comparar contra el algoritmo (validación manual).
    """
    os.makedirs(carpeta, exist_ok=True)
    tracker.reset()

    cap = cv2.VideoCapture(fuente)
    frame_idx = 0
    guardados = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= frame_fin:
            break

        if frame_idx >= frame_inicio:
            keypoints, scores = tracker(frame)
            persona, _ = seleccionar_persona_principal(keypoints, scores)
            frame_dibujado = (
                draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
                if persona is not None else frame
            )
            cv2.imwrite(f"{carpeta}/frame_{frame_idx:04d}.jpg", frame_dibujado)
            guardados += 1

        frame_idx += 1

    cap.release()
    print(f"Guardados {guardados} frames en la carpeta '{carpeta}/'")


def exportar_clip_alerta(fuente, tracker, frame_inicio, frame_fin, ruta_salida,
                          fps=30, margen_seg=1.0):
    """
    Guarda un clip corto de VIDEO (no imágenes sueltas), con el esqueleto
    dibujado, alrededor de una alerta detectada -- para que el usuario
    de la app VEA el instante exacto en vez de solo leer un timestamp.

    margen_seg: segundos extra antes/después del tramo, para dar contexto
    (ver la zancada completa, no solo el instante puntual de la alerta).
    """
    margen_frames = int(margen_seg * fps)
    inicio_real = max(0, frame_inicio - margen_frames)
    fin_real = frame_fin + margen_frames

    tracker.reset()
    cap = cv2.VideoCapture(fuente)

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # H.264 (avc1) es el que reproducen los navegadores directo en un
    # <video>; mp4v (el viejo códec MPEG-4) casi nunca lo reproduce
    # Chrome, aunque el archivo se genere sin error. Probamos avc1
    # primero, y si el build de OpenCV no lo tiene disponible, caemos a
    # mp4v (el clip va a quedar, pero solo se va a poder ver con un
    # reproductor local tipo VLC, no en el navegador).
    writer = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*"avc1"), fps, (ancho, alto))
    if not writer.isOpened():
        writer = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
        print("Aviso: este OpenCV no tiene codificador H.264 (avc1) -- "
              "el clip se guardó con mp4v y puede no reproducirse en el navegador.")

    frame_idx = 0
    guardados = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= fin_real:
            break

        if frame_idx >= inicio_real:
            keypoints, scores = tracker(frame)
            persona, _ = seleccionar_persona_principal(keypoints, scores)
            frame_out = (
                draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.5)
                if persona is not None else frame
            )
            writer.write(frame_out)
            guardados += 1

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Clip guardado en {ruta_salida} ({guardados} frames, "
          f"{guardados / fps:.1f}s con margen incluido)")


# ============================================================
# 2. Preprocesamiento: NaN, suavizado (OneEuroFilter), buffer
# ============================================================
def interpolar_nans(keypoints):
    """Rellena frames sin detección (NaN) interpolando linealmente, punto por punto."""
    N, K, _ = keypoints.shape
    flat = keypoints.reshape(N, -1).copy()

    for col in range(flat.shape[1]):
        serie = flat[:, col]
        nan_mask = np.isnan(serie)
        if nan_mask.any() and not nan_mask.all():
            indices_validos = np.flatnonzero(~nan_mask)
            indices_nan = np.flatnonzero(nan_mask)
            serie[nan_mask] = np.interp(indices_nan, indices_validos, serie[indices_validos])
        flat[:, col] = serie

    return flat.reshape(N, K, 2)


class OneEuroFilter:
    """
    Filtro adaptativo estándar para señales de tracking/pose (Casiez et
    al., 2012). Suaviza el ruido en reposo pero reacciona rápido a
    movimientos bruscos reales (como el impacto de un pie).
    """

    def __init__(self, freq, mincutoff=1.0, beta=0.3, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        a_d = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def suavizar_keypoints(keypoints, fps=30, mincutoff=1.0, beta=0.3):
    """Aplica un OneEuroFilter independiente a cada una de las 34 coordenadas."""
    n_frames, n_kpts, _ = keypoints.shape
    filtros = [[OneEuroFilter(freq=fps, mincutoff=mincutoff, beta=beta) for _ in range(2)]
               for _ in range(n_kpts)]

    suavizado = np.zeros_like(keypoints)
    for i in range(n_frames):
        for k in range(n_kpts):
            suavizado[i, k, 0] = filtros[k][0](keypoints[i, k, 0])
            suavizado[i, k, 1] = filtros[k][1](keypoints[i, k, 1])
    return suavizado


def construir_buffer(keypoints, window_size=30, stride=1):
    """
    Convierte un array continuo de keypoints (N, 17, 2) en ventanas
    deslizantes (num_ventanas, window_size, 34).
    """
    N = keypoints.shape[0]
    flat = keypoints.reshape(N, -1)

    if N < window_size:
        raise ValueError(f"El video tiene {N} frames, menos que window_size={window_size}")

    ventanas = []
    for start in range(0, N - window_size + 1, stride):
        ventanas.append(flat[start:start + window_size])

    return np.array(ventanas)


# ============================================================
# 3. Fórmulas clásicas de biomecánica (postura, cadencia, impacto, asimetría)
# ============================================================
def calcular_postura(keypoints):
    """Ángulo del tronco respecto a la vertical, por frame (grados). 0° = tronco vertical."""
    hombro_medio = (keypoints[:, LEFT_SHOULDER] + keypoints[:, RIGHT_SHOULDER]) / 2
    cadera_medio = (keypoints[:, LEFT_HIP] + keypoints[:, RIGHT_HIP]) / 2
    vector_tronco = hombro_medio - cadera_medio

    # Vertical de referencia = (0, -1): Y crece hacia abajo en la imagen
    return np.degrees(np.arctan2(vector_tronco[:, 0], -vector_tronco[:, 1]))


def calcular_inclinacion_cabeza(keypoints):
    """
    Ángulo cabeza-hombro respecto a la vertical, por frame (grados).
    Mismo criterio que calcular_postura, pero con NOSE en vez de la cadera
    como punto superior: nariz vs. punto medio de hombros. 0° = cabeza
    alineada verticalmente sobre los hombros; valores grandes indican
    cabeza agachada o proyectada hacia adelante/atrás (alineación
    craneo-cervical, ver objetivo general del documento).
    """
    nariz = keypoints[:, NOSE]
    hombro_medio = (keypoints[:, LEFT_SHOULDER] + keypoints[:, RIGHT_SHOULDER]) / 2
    vector_cabeza = nariz - hombro_medio

    return np.degrees(np.arctan2(vector_cabeza[:, 0], -vector_cabeza[:, 1]))


def detectar_contactos(y_tobillo, distancia_min=6, prominencia_min=3.0):
    """
    Detecta los frames de contacto con el suelo a partir de la posición
    vertical del tobillo (ya suavizada, relativa a la cadera).
    """
    contactos = []
    n = len(y_tobillo)
    for i in range(1, n - 1):
        if y_tobillo[i] > y_tobillo[i - 1] and y_tobillo[i] >= y_tobillo[i + 1]:
            izq = y_tobillo[max(0, i - distancia_min):i]
            der = y_tobillo[i:min(n, i + distancia_min)]
            valle = min(
                izq.min() if len(izq) else y_tobillo[i],
                der.min() if len(der) else y_tobillo[i],
            )
            prominencia = y_tobillo[i] - valle
            if prominencia >= prominencia_min:
                if not contactos or (i - contactos[-1]) >= distancia_min:
                    contactos.append(i)
    return contactos


def calcular_escala_corporal(keypoints):
    """
    Longitud de pierna (cadera-tobillo) en píxeles, mediana sobre todo
    el video -- referencia de "qué tan grande se ve la persona" en ESE
    video en particular. Se usa para normalizar umbrales de detección
    (en vez de un valor fijo en píxeles que no transfiere bien entre
    videos grabados a distinta distancia de cámara/resolución).
    """
    cadera = (keypoints[:, LEFT_HIP] + keypoints[:, RIGHT_HIP]) / 2
    tobillo = (keypoints[:, LEFT_ANKLE] + keypoints[:, RIGHT_ANKLE]) / 2
    return float(np.median(np.linalg.norm(cadera - tobillo, axis=1)))


def calcular_metricas_paso(keypoints, fps=30, es_corriendo=None, fraccion_prominencia=0.05,
                            distancia_min_seg=0.45):
    """
    A partir de los tobillos (relativos a la cadera, para ser inmune al
    paneo de cámara), calcula cadencia, impacto y asimetría.

    fraccion_prominencia: prominencia mínima para contar un contacto,
    como FRACCIÓN de la longitud de pierna de la persona en ESTE video
    (no un valor fijo en píxeles) -- así el umbral se adapta solo a la
    distancia de cámara/resolución de cada video.

    distancia_min_seg: separación mínima, en SEGUNDOS reales, entre dos
    contactos del mismo pie -- ningún pie toca el piso dos veces en menos
    de esto, ni corriendo muy rápido. Antes esto quedaba fijo en 6
    FRAMES sin importar los fps del video (bug: con distintos fps, 6
    frames son duraciones reales distintas); ahora se ata a segundos
    reales y se convierte a frames según el fps de este video. 0.45s
    fue el valor que separó limpio los contactos reales de los "hombros"
    falsos de la trayectoria del tobillo, en la validación visual hecha
    sobre un video de prueba. Si ves conteos raros en un video nuevo,
    repetí esa validación (graficar y_tobillo con los contactos marcados)
    antes de asumir que este valor sirve para cualquier cámara o ritmo.

    es_corriendo: array booleano opcional, indexado por FRAME (misma
    longitud aproximada que keypoints -- una aproximación razonable es
    pasar la clasificación por ventana de clasificar_actividad_ventana,
    ya que con stride=1 el índice de ventana ~ índice de frame). Si se
    da, los contactos que caigan en tramos SIN correr (caminata) se
    excluyen del cálculo, para no contaminar cadencia/impacto/asimetría
    de una sesión larga con un patrón de marcha mezclado.
    """
    escala = calcular_escala_corporal(keypoints)
    prominencia_min = escala * fraccion_prominencia
    distancia_min = max(1, int(distancia_min_seg * fps))

    cadera_medio_y = (keypoints[:, LEFT_HIP, 1] + keypoints[:, RIGHT_HIP, 1]) / 2
    y_izq = keypoints[:, LEFT_ANKLE, 1] - cadera_medio_y
    y_der = keypoints[:, RIGHT_ANKLE, 1] - cadera_medio_y

    contactos_izq = detectar_contactos(y_izq, distancia_min=distancia_min, prominencia_min=prominencia_min)
    contactos_der = detectar_contactos(y_der, distancia_min=distancia_min, prominencia_min=prominencia_min)

    if es_corriendo is not None:
        contactos_izq = [c for c in contactos_izq if c < len(es_corriendo) and es_corriendo[c]]
        contactos_der = [c for c in contactos_der if c < len(es_corriendo) and es_corriendo[c]]
        duracion_seg = float(np.asarray(es_corriendo).sum()) / fps
    else:
        duracion_seg = len(keypoints) / fps

    total_contactos = len(contactos_izq) + len(contactos_der)
    cadencia = (total_contactos / duracion_seg) * 60 if duracion_seg > 0 else 0.0

    def deceleracion_en_contactos(y, contactos):
        valores = []
        for c in contactos:
            if 1 <= c < len(y) - 1:
                v_antes = y[c] - y[c - 1]
                v_despues = y[c + 1] - y[c]
                valores.append(abs(v_antes - v_despues))
        return valores

    impactos = (deceleracion_en_contactos(y_izq, contactos_izq) +
                deceleracion_en_contactos(y_der, contactos_der))
    impacto = float(np.mean(impactos)) if impactos else 0.0

    def intervalo_promedio(contactos):
        if len(contactos) < 2:
            return None
        return float(np.mean(np.diff(contactos)))

    intervalo_izq = intervalo_promedio(contactos_izq)
    intervalo_der = intervalo_promedio(contactos_der)

    if intervalo_izq and intervalo_der:
        asimetria = (abs(intervalo_izq - intervalo_der) /
                     ((intervalo_izq + intervalo_der) / 2) * 100)
    else:
        asimetria = None

    return {
        "cadencia": cadencia,
        "impacto": impacto,
        "asimetria": asimetria,
        "contactos_izq": contactos_izq,
        "contactos_der": contactos_der,
    }


def contactos_en_segmento(contactos, frame_inicio, frame_fin):
    """Cuenta cuántos contactos detectados caen en el rango [inicio, fin)."""
    return sum(1 for c in contactos if frame_inicio <= c < frame_fin)


def clasificar_pisada(keypoints, contactos, heel_idx, toe_idx, umbral_px=5.0):
    """
    Clasifica el patrón de pisada ("talon", "mediopie" o "antepie") en
    cada contacto, comparando qué parte del pie está más cerca del
    suelo (mayor Y en píxeles) en ese instante.

    Requiere keypoints en formato Halpe26 (BodyWithFeet), no COCO-17 --
    usa heel_idx/toe_idx = LEFT_HEEL/LEFT_BIG_TOE o RIGHT_HEEL/RIGHT_BIG_TOE
    según el pie.
    """
    resultados = []
    for c in contactos:
        y_talon = keypoints[c, heel_idx, 1]
        y_dedo = keypoints[c, toe_idx, 1]
        diferencia = float(y_talon - y_dedo)  # + = talón más cerca del piso

        if diferencia > umbral_px:
            patron = "talon"
        elif diferencia < -umbral_px:
            patron = "antepie"
        else:
            patron = "mediopie"

        resultados.append({"frame": c, "patron": patron, "diferencia_px": diferencia})

    return resultados


def resumen_patron_pisada(resultados_izq, resultados_der):
    """Cuenta cuántas veces se dio cada patrón, combinando ambos pies."""
    conteo = {}
    todos = [r["patron"] for r in resultados_izq] + [r["patron"] for r in resultados_der]
    for patron in todos:
        conteo[patron] = conteo.get(patron, 0) + 1
    total = len(todos)
    return {p: {"conteo": n, "pct": (n / total * 100) if total else 0.0} for p, n in conteo.items()}


# ============================================================
# 4. Fatiga clásica (sesión completa) + corrección postural personalizada
# ============================================================
def detectar_fatiga(serie_asimetria, serie_postura, ventanas_baseline=60,
                     ventanas_recientes=30, umbral_pct=15):
    """
    Compara el inicio (línea base) contra el estado más reciente de una
    sesión larga, para asimetría y postura. serie_*: un valor por
    ventana a lo largo de TODA la sesión.
    """
    baseline_asim = np.nanmean(serie_asimetria[:ventanas_baseline])
    actual_asim = np.nanmean(serie_asimetria[-ventanas_recientes:])
    cambio_asim = (abs(actual_asim - baseline_asim) / baseline_asim * 100
                   if baseline_asim else 0.0)

    baseline_post = np.nanmean(np.abs(serie_postura[:ventanas_baseline]))
    actual_post = np.nanmean(np.abs(serie_postura[-ventanas_recientes:]))
    cambio_post = (abs(actual_post - baseline_post) / baseline_post * 100
                   if baseline_post else 0.0)

    posible_fatiga = (cambio_asim > umbral_pct) or (cambio_post > umbral_pct)

    return {
        "posible_fatiga": posible_fatiga,
        "cambio_asimetria_pct": cambio_asim,
        "cambio_postura_pct": cambio_post,
    }


def calcular_linea_base_postura(postura_serie, duracion_baseline_seg=30, fps=30,
                                 umbral_valido=90.0, inicio_baseline_seg=0.0,
                                 frames_validos=None):
    """
    Línea base personal: media y desviación estándar de la postura,
    tomada en una ventana de la sesión.

    frames_validos: array booleano opcional (misma longitud que
    postura_serie), True donde hubo detección real de persona (antes de
    interpolar NaNs, ver procesar_video). Si se da, esos frames se
    excluyen del cálculo -- evita que un tramo sin nadie en cuadro
    (arranque de la grabación, por ejemplo) contamine la línea base con
    valores interpolados sin sentido físico.

    inicio_baseline_seg: segundo desde el cual arrancar a tomar la línea
    base (por defecto 0.0). Útil como control manual adicional si
    igual querés saltar un tramo específico.

    umbral_valido: valores de |postura| por encima de este umbral (en
    grados) se consideran fallos de detección, no postura real -- nadie
    corre con el tronco a 90°+ de la vertical. Defensa adicional, más
    liviana que frames_validos.
    """
    inicio_frame = int(inicio_baseline_seg * fps)
    fin_frame = min(inicio_frame + int(duracion_baseline_seg * fps), len(postura_serie))
    base = postura_serie[inicio_frame:fin_frame]

    mascara = np.abs(base) <= umbral_valido
    if frames_validos is not None:
        mascara &= frames_validos[inicio_frame:fin_frame]

    base_valida = base[mascara]
    if len(base_valida) == 0:
        base_valida = base  # si TODO el tramo es inválido, usar como está (mejor que nada)

    std = float(np.std(base_valida))
    if std < 0.5:
        print(f"⚠️  Línea base con desviación casi nula ({std:.4f}°) -- probablemente la "
              f"ventana de línea base (inicio_baseline_seg={inicio_baseline_seg}s, "
              f"{duracion_baseline_seg}s de duración) cae total o parcialmente sobre un "
              f"tramo sin detección real, relleno con un valor constante por interpolación. "
              f"Probá con un inicio_baseline_seg más alto (después de que termine ese tramo).")

    return {"media": float(np.mean(base_valida)), "std": std + 1e-6}


def detectar_desviaciones_posturales(postura_serie, linea_base, umbral_z=2.0,
                                      duracion_sostenida_seg=1.0, fps=30, umbral_valido=90.0,
                                      frames_validos=None, ventana_incidente_seg=2.0,
                                      umbral_grados_min=None):
    """
    Marca los tramos donde la postura se aleja de la línea base
    PERSONAL más de umbral_z desviaciones estándar, sostenido al menos
    duracion_sostenida_seg segundos.

    umbral_grados_min: además del z-score, exigí que la desviación real
    sea de al menos estos grados respecto a la media de tu línea base.
    Sin esto (None), alguien con una línea base muy estable (std chico)
    puede disparar alertas por cambios de apenas 3-4°, estadísticamente
    "raros para él" pero imperceptibles en la práctica. Con esto, un
    tramo solo cuenta si es raro Y además grande en grados de verdad.

    frames_validos: array booleano opcional (misma longitud que
    postura_serie), True donde hubo detección real de persona. Los
    frames sin detección se ignoran al buscar desviaciones (evita
    alertas falsas por huecos de tracking). Además, si una alerta
    termina justo antes de un tramo sin detección (dentro de
    ventana_incidente_seg), se etiqueta como "posible_incidente" en vez
    de desviación de técnica -- típico de una caída (salís del encuadre
    o la cámara pierde el tracking).

    umbral_valido: frames con |postura| por encima de este valor se
    ignoran al buscar desviaciones (fallo de detección, no fatiga o
    mala técnica real).
    """
    z_scores = (postura_serie - linea_base["media"]) / linea_base["std"]
    desvio_grados = postura_serie - linea_base["media"]
    frames_min = int(duracion_sostenida_seg * fps)
    frames_incidente = int(ventana_incidente_seg * fps)
    postura_valida = np.abs(postura_serie) <= umbral_valido
    if frames_validos is not None:
        postura_valida &= frames_validos

    def cerrar_alerta(inicio, fin):
        z_prom = float(np.mean(z_scores[inicio:fin]))
        delta_grados = float(np.mean(desvio_grados[inicio:fin]))
        posible_incidente = False
        if frames_validos is not None:
            ventana_post = frames_validos[fin:fin + frames_incidente]
            if len(ventana_post) > 0 and not ventana_post.any():
                posible_incidente = True
        return {
            "inicio_seg": inicio / fps,
            "fin_seg": fin / fps,
            "z_promedio": z_prom,
            "delta_grados": delta_grados,
            "direccion": "adelante" if z_prom > 0 else "atrás",
            "posible_incidente": posible_incidente,
        }

    alertas = []
    en_desviacion = False
    inicio = None

    for i, z in enumerate(z_scores):
        fuera_de_rango = abs(z) > umbral_z and postura_valida[i]
        if umbral_grados_min is not None:
            fuera_de_rango = fuera_de_rango and abs(desvio_grados[i]) >= umbral_grados_min
        if fuera_de_rango and not en_desviacion:
            en_desviacion, inicio = True, i
        elif not fuera_de_rango and en_desviacion:
            if i - inicio >= frames_min:
                alertas.append(cerrar_alerta(inicio, i))
            en_desviacion = False

    if en_desviacion and len(z_scores) - inicio >= frames_min:
        alertas.append(cerrar_alerta(inicio, len(z_scores)))

    return alertas


def detectar_desviaciones_adaptativas(postura_serie, ventana_seg=75, umbral_z=2.5,
                                       duracion_sostenida_seg=1.0, fps=30,
                                       frames_validos=None, umbral_valido=90.0,
                                       umbral_grados_min=None):
    """
    Alternativa a detectar_desviaciones_posturales para señales que
    DERIVAN naturalmente a lo largo de la sesión (por ejemplo, la
    inclinación de cabeza medida de perfil, que en la validación con
    dataBse3.MOV osciló entre ~30° y ~46° minuto a minuto, sin una
    tendencia clara -- una línea base fija, corta o larga, siempre
    termina marcando la mitad de la sesión como "anormal").

    En vez de una línea base fija tomada al principio, compara cada
    instante contra el promedio de los últimos ventana_seg segundos --
    una "línea base que se mueve con vos". Esto filtra la deriva lenta
    y solo marca cambios BRUSCOS respecto a tu tendencia reciente, que
    es lo que de verdad se parece a "algo exagerado", en vez de "distinto
    a como estabas hace 8 minutos".

    Ojo: los primeros/últimos ventana_seg/2 segundos tienen una línea
    base un poco menos precisa (se repite el valor del borde en vez de
    promediar datos reales de ambos lados) -- no es grave, pero no
    interpretes una alerta pegada al inicio o al final con la misma
    confianza que una del medio de la sesión.

    Devuelve alertas con el mismo formato que detectar_desviaciones_posturales
    (inicio_seg, fin_seg, z_promedio, delta_grados, direccion), así que
    generar_mensaje_alerta / la misma lógica de reporte sirve para ambas.

    umbral_grados_min: igual que en detectar_desviaciones_posturales --
    además del z-score, exigí una desviación real de al menos estos
    grados respecto a la tendencia reciente.
    """
    ventana_frames = max(1, int(ventana_seg * fps))
    pad = ventana_frames // 2
    serie_pad = np.pad(postura_serie, pad, mode="edge")
    kernel = np.ones(ventana_frames) / ventana_frames
    media_movil = np.convolve(serie_pad, kernel, mode="valid")[:len(postura_serie)]

    desvio = postura_serie - media_movil
    std_local = float(np.std(desvio)) + 1e-6
    z_scores = desvio / std_local

    frames_min = int(duracion_sostenida_seg * fps)
    postura_valida = np.abs(postura_serie) <= umbral_valido
    if frames_validos is not None:
        postura_valida &= frames_validos

    def cerrar_alerta(inicio, fin):
        z_prom = float(np.mean(z_scores[inicio:fin]))
        delta_grados = float(np.mean(desvio[inicio:fin]))
        return {
            "inicio_seg": inicio / fps,
            "fin_seg": fin / fps,
            "z_promedio": z_prom,
            "delta_grados": delta_grados,
            "direccion": "adelante" if delta_grados > 0 else "atrás",
            "posible_incidente": False,
        }

    alertas = []
    en_desviacion = False
    inicio = None
    for i, z in enumerate(z_scores):
        fuera_de_rango = abs(z) > umbral_z and postura_valida[i]
        if umbral_grados_min is not None:
            fuera_de_rango = fuera_de_rango and abs(desvio[i]) >= umbral_grados_min
        if fuera_de_rango and not en_desviacion:
            en_desviacion, inicio = True, i
        elif not fuera_de_rango and en_desviacion:
            if i - inicio >= frames_min:
                alertas.append(cerrar_alerta(inicio, i))
            en_desviacion = False

    if en_desviacion and len(z_scores) - inicio >= frames_min:
        alertas.append(cerrar_alerta(inicio, len(z_scores)))

    return alertas


def formatear_tiempo(segundos):
    """Segundos -> 'M:SS', para que las alertas se lean en minutos y no
    en segundos crudos (más fácil de ubicar en un video de varios minutos)."""
    minutos = int(segundos // 60)
    seg = int(round(segundos % 60))
    return f"{minutos}:{seg:02d}"


def generar_mensaje_alerta(alerta):
    inicio = formatear_tiempo(alerta["inicio_seg"])
    fin = formatear_tiempo(alerta["fin_seg"])
    if alerta.get("posible_incidente"):
        return (f"[{inicio}-{fin}] "
                f"Posible caída o incidente (desviación extrema seguida de pérdida de tracking, "
                f"z={alerta['z_promedio']:.1f}) -- no se interpreta como desviación de técnica")
    delta = alerta.get("delta_grados")
    detalle_grados = f", Δ={delta:+.1f}°" if delta is not None else ""
    return (f"[{inicio}-{fin}] "
            f"Tronco inclinado hacia {alerta['direccion']} más de lo normal para vos "
            f"(z={alerta['z_promedio']:.1f}{detalle_grados})")


# ============================================================
# 5. Etiquetas por ventana + procesamiento multi-video (para el Transformer)
# ============================================================
def generar_etiquetas_por_ventana(keypoints, window_size=30, stride=1, fps=30, contexto_seg=3.0):
    """
    Para cada ventana de 30 frames: postura e impacto instantáneos (solo
    esos 30 frames); cadencia y asimetría con una ventana de contexto
    más ancha (contexto_seg), centrada en el mismo punto temporal.
    """
    N = keypoints.shape[0]
    contexto_frames = int(contexto_seg * fps)
    etiquetas = {"impacto": [], "cadencia": [], "postura": [], "asimetria": []}

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        centro = (start + end) // 2

        postura_ventana = calcular_postura(keypoints[start:end])
        etiquetas["postura"].append(float(np.mean(postura_ventana)))

        ctx_inicio = max(0, centro - contexto_frames // 2)
        ctx_fin = min(N, centro + contexto_frames // 2)
        m = calcular_metricas_paso(keypoints[ctx_inicio:ctx_fin], fps=fps)

        etiquetas["cadencia"].append(m["cadencia"])
        etiquetas["impacto"].append(m["impacto"])
        etiquetas["asimetria"].append(m["asimetria"] if m["asimetria"] is not None else np.nan)

    return {k: np.array(v) for k, v in etiquetas.items()}


def procesar_multiples_videos(rutas_videos, tracker, window_size=30, stride=1, fps=30,
                               contexto_seg=3.0, carpeta_cache="cache_keypoints"):
    """
    Corre extracción -> interpolación -> suavizado -> buffer -> etiquetas
    para cada video, y devuelve todo concatenado + de qué video vino
    cada ventana. Cachea los keypoints crudos en disco (RTMPose es lo
    lento; si ya está cacheado, no se vuelve a correr).
    """
    os.makedirs(carpeta_cache, exist_ok=True)

    buffers = []
    etq_por_metrica = {"impacto": [], "cadencia": [], "postura": [], "asimetria": []}
    video_id_por_ventana = []

    for video_idx, ruta in enumerate(rutas_videos):
        nombre_cache = os.path.join(carpeta_cache, os.path.basename(ruta) + ".npy")

        if os.path.exists(nombre_cache):
            print(f"Video {video_idx}: usando cache ({nombre_cache})")
            keypoints = np.load(nombre_cache)
        else:
            print(f"Video {video_idx}: procesando con RTMPose ({ruta}) -- puede tardar varios minutos...")
            keypoints, total_frames, _, _ = procesar_video(ruta, tracker)
            np.save(nombre_cache, keypoints)
            print(f"  -> cache guardado en {nombre_cache}")

        keypoints_i = interpolar_nans(keypoints)
        keypoints_s = suavizar_keypoints(keypoints_i, fps=fps)

        buf = construir_buffer(keypoints_s, window_size=window_size, stride=stride)
        etq = generar_etiquetas_por_ventana(keypoints_s, window_size=window_size,
                                             stride=stride, fps=fps, contexto_seg=contexto_seg)

        buffers.append(buf)
        for k in etq_por_metrica:
            etq_por_metrica[k].append(etq[k])
        video_id_por_ventana.extend([video_idx] * len(buf))
        print(f"  -> {len(buf)} ventanas")

    buffer_total = np.concatenate(buffers, axis=0)
    etiquetas_total = {k: np.concatenate(v) for k, v in etq_por_metrica.items()}
    return buffer_total, etiquetas_total, np.array(video_id_por_ventana)


def filtrar_validos(indices, etiquetas_dict):
    """Devuelve solo los índices donde TODAS las etiquetas del dict son válidas (no NaN)."""
    validas = np.ones(len(indices), dtype=bool)
    for v in etiquetas_dict.values():
        validas &= ~np.isnan(v[indices])
    return indices[validas]


def split_por_video(video_id, etiquetas_total, fraccion_val=0.2, seed=42):
    """Separa entrenamiento/validación por VIDEO completo (no por ventana), para no filtrar datos."""
    videos_unicos = np.unique(video_id)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(videos_unicos) * fraccion_val))
    videos_val = rng.choice(videos_unicos, size=n_val, replace=False)

    es_val = np.isin(video_id, videos_val)
    idx_train = filtrar_validos(np.where(~es_val)[0], etiquetas_total)
    idx_val = filtrar_validos(np.where(es_val)[0], etiquetas_total)

    return idx_train, idx_val, videos_val


# ============================================================
# 6. Arquitectura Transformer (regresión de las 4 métricas)
# ============================================================
class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal estándar (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """
    Encoder-only Transformer para diagnóstico biomecánico.
    Entrada: (batch, 30, 34) -> 4 métricas de salida.
    """

    def __init__(self, input_dim=34, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, window_size=30):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_impacto = nn.Linear(d_model, 1)
        self.head_cadencia = nn.Linear(d_model, 1)
        self.head_postura = nn.Linear(d_model, 1)
        self.head_asimetria = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        resumen = x.mean(dim=1)
        return {
            "impacto": self.head_impacto(resumen).squeeze(-1),
            "cadencia": self.head_cadencia(resumen).squeeze(-1),
            "postura": self.head_postura(resumen).squeeze(-1),
            "asimetria": self.head_asimetria(resumen).squeeze(-1),
        }


class Normalizador:
    """Normaliza cada etiqueta a media 0, desviación 1 (calculado sobre train)."""

    def __init__(self, etiquetas_dict):
        self.medias = {k: float(np.mean(v)) for k, v in etiquetas_dict.items()}
        self.stds = {k: float(np.std(v)) + 1e-6 for k, v in etiquetas_dict.items()}

    def normalizar(self, etiquetas_dict):
        return {k: (v - self.medias[k]) / self.stds[k] for k, v in etiquetas_dict.items()}

    def desnormalizar(self, valor, nombre):
        return valor * self.stds[nombre] + self.medias[nombre]


class VentanasDataset(Dataset):
    """Empareja cada ventana del buffer con sus 4 etiquetas ya normalizadas."""

    def __init__(self, buffer, etiquetas_normalizadas, indices):
        self.buffer = buffer[indices]
        self.etiquetas = {k: v[indices] for k, v in etiquetas_normalizadas.items()}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        x = torch.tensor(self.buffer[idx], dtype=torch.float32)
        y = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.etiquetas.items()}
        return x, y


def entrenar(modelo, dataloader_train, dataloader_val, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()

    for epoch in range(epochs):
        modelo.train()
        perdida_train = 0.0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            salidas = modelo(x)
            perdida = sum(criterio(salidas[k], y[k]) for k in salidas)
            perdida.backward()
            optimizer.step()
            perdida_train += perdida.item()

        modelo.eval()
        perdida_val = 0.0
        with torch.no_grad():
            for x, y in dataloader_val:
                salidas = modelo(x)
                perdida = sum(criterio(salidas[k], y[k]) for k in salidas)
                perdida_val += perdida.item()

        n_train_batches = max(len(dataloader_train), 1)
        n_val_batches = max(len(dataloader_val), 1)
        print(f"Epoch {epoch+1}/{epochs} -- train: {perdida_train/n_train_batches:.4f}"
              f" -- val: {perdida_val/n_val_batches:.4f}")

    return modelo


def evaluar_fidelidad(modelo, dataloader_val, normalizador):
    """MAE, RMSE, R² y correlación por métrica -- los números para la tesis."""
    modelo.eval()
    predicciones = {k: [] for k in ["impacto", "cadencia", "postura", "asimetria"]}
    reales = {k: [] for k in ["impacto", "cadencia", "postura", "asimetria"]}

    with torch.no_grad():
        for x, y in dataloader_val:
            salidas = modelo(x)
            for k in predicciones:
                predicciones[k].extend(normalizador.desnormalizar(salidas[k].numpy(), k))
                reales[k].extend(normalizador.desnormalizar(y[k].numpy(), k))

    unidades = {"impacto": "px/frame²", "cadencia": "pasos/min", "postura": "°", "asimetria": "%"}
    resultados = {}

    print("--- Fidelidad en validación ---")
    for k in predicciones:
        pred = np.array(predicciones[k])
        real = np.array(reales[k])

        mae = np.mean(np.abs(pred - real))
        rmse = np.sqrt(np.mean((pred - real) ** 2))
        ss_res = np.sum((real - pred) ** 2)
        ss_tot = np.sum((real - real.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r = np.corrcoef(pred, real)[0, 1] if len(pred) > 1 else float("nan")

        print(f"\n{k}:")
        print(f"  MAE:  {mae:.2f} {unidades[k]}")
        print(f"  RMSE: {rmse:.2f} {unidades[k]}")
        print(f"  R²:   {r2:.3f}")
        print(f"  Correlación (r): {r:.3f}")

        resultados[k] = {"pred": pred, "real": real, "mae": mae, "rmse": rmse, "r2": r2, "r": r}

    return resultados


# ============================================================
# 7. Autoencoder Transformer (fatiga / detección de anomalías)
# ============================================================
class TransformerAutoencoder(nn.Module):
    """
    Encoder: comprime la ventana (30, 34) a un vector latente chico.
    Decoder: reconstruye la ventana a partir de ese vector.
    Error de reconstrucción alto = esta ventana no se parece al
    movimiento "normal" que la red aprendió.
    """

    def __init__(self, input_dim=34, d_model=64, nhead=4, num_layers=1,
                 dim_feedforward=128, dropout=0.1, window_size=30, latent_dim=16):
        super().__init__()
        self.window_size = window_size

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size)
        capa_enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(capa_enc, num_layers=num_layers)
        self.a_latente = nn.Linear(d_model, latent_dim)

        self.de_latente = nn.Linear(latent_dim, d_model)
        self.pos_encoding_dec = PositionalEncoding(d_model, max_len=window_size)
        capa_dec = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(capa_dec, num_layers=num_layers)
        self.salida = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.embedding(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        resumen = h.mean(dim=1)
        z = self.a_latente(resumen)

        h_dec = self.de_latente(z).unsqueeze(1).repeat(1, self.window_size, 1)
        h_dec = self.pos_encoding_dec(h_dec)
        h_dec = self.decoder(h_dec)
        return self.salida(h_dec)


class VentanasAutoencoderDataset(Dataset):
    """Dataset no supervisado: la entrada ES el objetivo (reconstruirse a sí misma)."""

    def __init__(self, buffer, indices):
        self.buffer = buffer[indices]

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        x = torch.tensor(self.buffer[idx], dtype=torch.float32)
        return x, x


def entrenar_autoencoder(modelo, dataloader_train, dataloader_val, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()

    mejor_val = float("inf")
    mejor_estado = None

    for epoch in range(epochs):
        modelo.train()
        perdida_train = 0.0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            reconstruccion = modelo(x)
            perdida = criterio(reconstruccion, y)
            perdida.backward()
            optimizer.step()
            perdida_train += perdida.item()

        modelo.eval()
        perdida_val = 0.0
        with torch.no_grad():
            for x, y in dataloader_val:
                perdida_val += criterio(modelo(x), y).item()

        perdida_train /= max(len(dataloader_train), 1)
        perdida_val /= max(len(dataloader_val), 1)

        if perdida_val < mejor_val:
            mejor_val = perdida_val
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}

        print(f"Epoch {epoch+1}/{epochs} -- train: {perdida_train:.4f} -- val: {perdida_val:.4f}")

    modelo.load_state_dict(mejor_estado)  # el MEJOR checkpoint, no el último
    print(f"\nMejor val loss: {mejor_val:.4f} -- checkpoint restaurado")
    return modelo


def calcular_error_reconstruccion(modelo, buffer):
    """Error de reconstrucción (MSE) por ventana."""
    modelo.eval()
    x = torch.tensor(buffer, dtype=torch.float32)
    with torch.no_grad():
        reconstruccion = modelo(x)
        error = ((reconstruccion - x) ** 2).mean(dim=(1, 2)).numpy()
    return error


def detectar_anomalias(error_reconstruccion, umbral_z=2.0, duracion_sostenida_ventanas=3):
    """Igual que detectar_desviaciones_posturales, pero para error de reconstrucción (siempre positivo)."""
    media = float(np.mean(error_reconstruccion))
    std = float(np.std(error_reconstruccion)) + 1e-6
    z_scores = (error_reconstruccion - media) / std

    alertas = []
    en_anomalia = False
    inicio = None

    for i, z in enumerate(z_scores):
        alto = z > umbral_z
        if alto and not en_anomalia:
            en_anomalia, inicio = True, i
        elif not alto and en_anomalia:
            if i - inicio >= duracion_sostenida_ventanas:
                alertas.append({"ventana_inicio": inicio, "ventana_fin": i,
                                 "z_promedio": float(np.mean(z_scores[inicio:i]))})
            en_anomalia = False

    if en_anomalia and len(z_scores) - inicio >= duracion_sostenida_ventanas:
        alertas.append({"ventana_inicio": inicio, "ventana_fin": len(z_scores),
                         "z_promedio": float(np.mean(z_scores[inicio:]))})

    return alertas, media, std


def filtrar_alertas_por_deteccion(alertas, frames_validos, umbral_deteccion=0.5):
    """
    Descarta alertas que caen mayormente en tramos SIN detección real de
    persona (ej. el inicio del video antes de subirse a la cinta) --
    tras interpolar esos huecos, el error de reconstrucción puede salir
    alto sin que sea fatiga real, solo un hueco de datos.

    frames_validos: array booleano (N,), True donde hubo detección real
    ANTES de interpolar. umbral_deteccion: fracción mínima de frames con
    detección real dentro del rango de la alerta para conservarla.
    """
    alertas_filtradas = []
    for alerta in alertas:
        inicio, fin = alerta["ventana_inicio"], alerta["ventana_fin"]
        fraccion = frames_validos[inicio:fin].mean() if fin > inicio else 0.0
        if fraccion >= umbral_deteccion:
            alertas_filtradas.append(alerta)
    return alertas_filtradas


def clasificar_actividad_ventana(cadencia_por_ventana, umbral_caminata=140.0):
    """
    Clasifica cada ventana como "corriendo" (True) o "caminando" (False),
    usando un umbral simple sobre la cadencia (pasos/min) -- la
    transición caminar/trotar ocurre, según la literatura, alrededor de
    esa cifra. No es un valor exacto validado para vos -- ajustalo si
    ves que clasifica mal en tu caso.
    """
    return np.asarray(cadencia_por_ventana) >= umbral_caminata


def entrenar_autoencoder_fatiga(buffer_sesion, minutos_baseline=3, fps=30,
                                 window_size=30, stride=1, epochs=30, lr=1e-3, batch_size=16,
                                 mascara_valida=None):
    """
    Entrena el autoencoder SOLO con los primeros minutos de una sesión
    (corredor fresco).

    mascara_valida: array booleano opcional (misma longitud que
    buffer_sesion), True donde la ventana debe considerarse válida para
    entrenar (ej. clasificada como "corriendo", no "caminando" -- ver
    clasificar_actividad_ventana). Si se da, se descartan del tramo de
    línea base las ventanas que no cumplan, para no contaminar la
    definición de "normal" con un gesto de caminata.
    """
    frames_baseline = int(minutos_baseline * 60 * fps)
    num_ventanas_baseline = max(1, (frames_baseline - window_size) // stride + 1)
    num_ventanas_baseline = min(num_ventanas_baseline, len(buffer_sesion))

    indices_baseline = np.arange(num_ventanas_baseline)

    if mascara_valida is not None:
        n_antes = len(indices_baseline)
        indices_baseline = indices_baseline[mascara_valida[indices_baseline]]
        print(f"Línea base: {n_antes} ventanas -> {len(indices_baseline)} tras descartar "
              f"tramos clasificados como caminata")

    n_val = max(1, int(len(indices_baseline) * 0.2))
    idx_train_base = indices_baseline[:-n_val]
    idx_val_base = indices_baseline[-n_val:]

    dataset_train = VentanasAutoencoderDataset(buffer_sesion, idx_train_base)
    dataset_val = VentanasAutoencoderDataset(buffer_sesion, idx_val_base)
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    print(f"Entrenando con {len(idx_train_base)} ventanas de línea base "
          f"({minutos_baseline} min iniciales), validando con {len(idx_val_base)}")

    modelo = TransformerAutoencoder()
    modelo = entrenar_autoencoder(modelo, dl_train, dl_val, epochs=epochs, lr=lr)

    return modelo, num_ventanas_baseline


def perfil_fatiga_sesion(modelo, buffer_sesion, fps=30, window_size=30, stride=1):
    """Error de reconstrucción para cada ventana de la sesión, en orden temporal."""
    error = calcular_error_reconstruccion(modelo, buffer_sesion)
    tiempos_seg = (np.arange(len(error)) * stride + window_size / 2) / fps
    return tiempos_seg, error


def _cargar_o_procesar_sesion(ruta, carpeta_cache):
    """Helper interno: extrae keypoints crudos de un video, cacheando en disco por nombre."""
    os.makedirs(carpeta_cache, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    archivo_cache = os.path.join(carpeta_cache, nombre + "_raw.npy")
    archivo_fps = os.path.join(carpeta_cache, nombre + "_fps.npy")

    if os.path.exists(archivo_cache) and os.path.exists(archivo_fps):
        return nombre, np.load(archivo_cache), float(np.load(archivo_fps))

    tracker = crear_tracker(mode="balanced", backend="onnxruntime", device="cpu")
    keypoints, _, _, fps = procesar_video(ruta, tracker)
    np.save(archivo_cache, keypoints)
    np.save(archivo_fps, np.array(fps))
    return nombre, keypoints, fps


def entrenar_autoencoder_fatiga_general(rutas_sesiones, minutos_baseline=3,
                                         window_size=30, stride=1, fraccion_val=0.2,
                                         epochs=30, lr=1e-3, batch_size=16, seed=42,
                                         carpeta_cache="cache_keypoints"):
    """
    Entrena UN modelo de fatiga combinando el tramo fresco (primeros
    minutos_baseline) de VARIAS sesiones -- en vez de un modelo nuevo
    por sesión, este se entrena una sola vez y se reutiliza en sesiones
    futuras sin re-entrenar.

    Separa por SESIÓN COMPLETA en train/val (no por ventana), igual que
    en el resto del proyecto, para no filtrar datos.

    rutas_sesiones: lista de rutas de video (mínimo recomendado: 8-10
    sesiones largas distintas).

    Devuelve: (modelo, info_sesiones) -- info_sesiones tiene nombre/fps/
    cantidad de ventanas de cada sesión usada.
    """
    buffers_baseline = []
    sesion_id_por_ventana = []
    info_sesiones = []

    for sesion_idx, ruta in enumerate(rutas_sesiones):
        print(f"Sesión {sesion_idx}: {ruta}")
        nombre, keypoints, fps = _cargar_o_procesar_sesion(ruta, carpeta_cache)

        keypoints_interp = interpolar_nans(keypoints)
        keypoints_suavizados = suavizar_keypoints(keypoints_interp, fps=fps)

        frames_baseline = min(int(minutos_baseline * 60 * fps), len(keypoints_suavizados))
        keypoints_baseline = keypoints_suavizados[:frames_baseline]

        if len(keypoints_baseline) < window_size:
            print(f"  -> descartada: muy corta para {minutos_baseline} min de línea base")
            continue

        buffer_baseline = construir_buffer(keypoints_baseline, window_size=window_size, stride=stride)
        buffers_baseline.append(buffer_baseline)
        sesion_id_por_ventana.extend([sesion_idx] * len(buffer_baseline))
        info_sesiones.append({"idx": sesion_idx, "nombre": nombre, "ruta": ruta, "fps": fps,
                               "n_ventanas_baseline": len(buffer_baseline)})
        print(f"  -> {len(buffer_baseline)} ventanas de línea base (fps={fps:.1f})")

    buffer_baseline_total = np.concatenate(buffers_baseline, axis=0)
    sesion_id_por_ventana = np.array(sesion_id_por_ventana)

    sesiones_unicas = np.unique(sesion_id_por_ventana)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(sesiones_unicas) * fraccion_val))
    sesiones_val = rng.choice(sesiones_unicas, size=n_val, replace=False)

    es_val = np.isin(sesion_id_por_ventana, sesiones_val)
    idx_train = np.where(~es_val)[0]
    idx_val = np.where(es_val)[0]

    print(f"\nTotal ventanas de línea base: {len(buffer_baseline_total)}")
    print(f"Train: {len(idx_train)} ventanas de {len(sesiones_unicas) - n_val} sesiones")
    print(f"Val:   {len(idx_val)} ventanas de {n_val} sesiones ({sesiones_val})")

    dataset_train = VentanasAutoencoderDataset(buffer_baseline_total, idx_train)
    dataset_val = VentanasAutoencoderDataset(buffer_baseline_total, idx_val)
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    modelo = TransformerAutoencoder()
    modelo = entrenar_autoencoder(modelo, dl_train, dl_val, epochs=epochs, lr=lr)

    return modelo, info_sesiones


def validar_fatiga_multisesion(modelo, rutas_sesiones, window_size=30, stride=1,
                                carpeta_cache="cache_keypoints"):
    """
    Aplica el modelo YA ENTRENADO a la sesión COMPLETA de cada video (no
    solo el tramo de línea base) -- la validación real de que el modelo
    generaliza entre sesiones, no solo memorizó una.

    Devuelve un dict {nombre_sesion: {"tiempos": ..., "error": ...}}.
    """
    resultados = {}
    for ruta in rutas_sesiones:
        nombre, keypoints, fps = _cargar_o_procesar_sesion(ruta, carpeta_cache)

        keypoints_interp = interpolar_nans(keypoints)
        keypoints_suavizados = suavizar_keypoints(keypoints_interp, fps=fps)
        buffer_sesion = construir_buffer(keypoints_suavizados, window_size=window_size, stride=stride)

        tiempos, error = perfil_fatiga_sesion(modelo, buffer_sesion, fps=fps,
                                               window_size=window_size, stride=stride)
        resultados[nombre] = {"tiempos": tiempos, "error": error}
        print(f"{nombre}: {len(error)} ventanas procesadas")

    return resultados


def suavizar_serie(serie, ventana=90):
    """
    Promedio móvil centrado -- útil para ver la TENDENCIA de una señal
    ruidosa (como el error de reconstrucción crudo) sin perder de vista
    en qué momento ocurre el cambio real.
    """
    serie = np.asarray(serie, dtype=float)
    n = len(serie)
    salida = np.empty(n)
    mitad = ventana // 2
    for i in range(n):
        inicio = max(0, i - mitad)
        fin = min(n, i + mitad + 1)
        salida[i] = serie[inicio:fin].mean()
    return salida
