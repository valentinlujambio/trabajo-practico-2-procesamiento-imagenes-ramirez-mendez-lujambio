import cv2
import numpy as np
import sys
import os

from help_show import imshow
from caracteres import _detectar_chars


def _dibujar_cajas(img, cajas, color=(0, 255, 0), grosor=1):
    """Devuelve una copia de img con las cajas dibujadas (solo para debug)."""
    vis = img.copy()
    for (x, y, w, h) in cajas:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, grosor)
    return vis


def _hay_azul_arriba(hsv, fila, ch):
    """Corroboramos si hay franja azul arriba de la fila de caracteres"""
    xs0 = min(b[0] for b in fila); xs1 = max(b[0] + b[2] for b in fila)
    ys0 = min(b[1] for b in fila)
    y0 = max(0, int(ys0 - 1.1 * ch)); y1 = max(0, int(ys0 - 0.1 * ch))
    if y1 <= y0:
        return False
    band = hsv[y0:y1, xs0:xs1]
    if band.size == 0:
        return False
    Hh, Ss, Vv = band[..., 0], band[..., 1], band[..., 2]
    blue = ((Hh >= 100) & (Hh <= 135) & (Ss >= 60) & (Vv >= 40))
    return blue.mean() > 0.20


def _blancura_fondo(hsv, fila):
    """ Buscamos el fondo blanco de la chapa para descartar carteles """
    xs0 = min(b[0] for b in fila); xs1 = max(b[0] + b[2] for b in fila)
    ys0 = min(b[1] for b in fila); ys1 = max(b[1] + b[3] for b in fila)
    crop = hsv[ys0:ys1, xs0:xs1]
    if crop.size == 0:
        return 0.0
    white = (crop[..., 2] > 140) & (crop[..., 1] < 70)
    return float(white.mean())


def _ccs_caracteres_like(th, Hp):
    """Buscamos los caracteres en una binaria de toda la imagen."""
    num, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    cajas = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Sacamos los caracteres que no son enormes
        if h < 10 or h > 0.30 * Hp:
            continue
        ar = w / float(h)
        # Sacamos los caracteres que no son altos y angostos
        if ar < 0.15 or ar > 0.90:
            continue
        fill = a / float(w * h)
        if fill < 0.15 or fill > 0.97:
            continue
        cajas.append((x, y, w, h))
    return cajas


def _score_fila(run, ch):
    """Esto es importante porque matcheamos con la geometría de la patente:
       - n caracteres ≈ 7
       - ancho del conjunto ≈ 5.85 * alto_char (2L+3N+2L + separaciones)
       - ancho medio de char ≈ 0.55 * alto
       - espaciado regular entre centros
    Devolvemos score o None si no es aceptable."""
    n = len(run)
    if n < 5 or n > 9:
        return None
    xs0 = min(b[0] for b in run); xs1 = max(b[0] + b[2] for b in run)
    row_w = xs1 - xs0
    row_aspect = row_w / float(ch) # objetivo ~5.85 para 7 caracteres
    wmed = float(np.median([b[2] for b in run]))
    w_ratio = wmed / ch # objetivo ~0.55
    centros = sorted([b[0] + b[2] / 2.0 for b in run])
    pitches = np.diff(centros)
    pitch_cv = float(np.std(pitches) / (np.mean(pitches) + 1e-6)) if len(pitches) else 1.0

    score = 0.0
    score += -1.5 * abs(n - 7)
    score += -1.0 * abs(row_aspect - 5.85)
    score += -3.0 * abs(w_ratio - 0.55)
    score += -2.0 * pitch_cv
    return score


def _evaluar_filas(cajas):
    """Agrupa cajas en filas (altura + cy similares), las parte en runs por
    separación grande, y puntúa cada run. Devolvemos [(run, score), ...]."""
    resultados = []
    for seed in cajas:
        sh = seed[3]; scy = seed[1] + seed[3] / 2.0
        grupo = [b for b in cajas
                 if 0.7 * sh <= b[3] <= 1.45 * sh
                 and abs((b[1] + b[3] / 2.0) - scy) <= 0.5 * sh]
        if len(grupo) < 4:
            continue
        grupo.sort(key=lambda b: b[0])
        ch = float(np.median([b[3] for b in grupo]))
        # partir en runs por gap grande entre cajas consecutivas
        runs = [[grupo[0]]]
        for prev, cur in zip(grupo, grupo[1:]):
            gap = cur[0] - (prev[0] + prev[2])
            if gap > 1.3 * ch:
                runs.append([cur])
            else:
                runs[-1].append(cur)
        for run in runs:
            sc = _score_fila(run, ch)
            if sc is not None:
                resultados.append((run, sc))
    return resultados


def _buscar_fila_patente(img, debug=False):
    """Busca en toda la imagen la fila de ~7 caracteres que mejor matchea la
    geometría de una patente Mercosur. Devolvemos la lista de cajas o []."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    Hp = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if debug:
        imshow(gray, title="2 - Escala de grises", blocking=True)
        imshow(gray_eq, title="3 - CLAHE (mejora de contraste local)", blocking=True)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mejor_fila, mejor_score = [], -float('inf')
    for idx, (bs, C) in enumerate([(21, 9), (31, 12), (15, 6)]):
        th = cv2.adaptiveThreshold(
            gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, bs, C)
        if debug:
            imshow(th, title=f"4.{idx + 1} - Umbral adaptativo (bs={bs}, C={C})", blocking=True)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
        if debug:
            imshow(th, title=f"5.{idx + 1} - Apertura morfológica (bs={bs})", blocking=True)
        cajas = _ccs_caracteres_like(th, Hp)
        if debug:
            vis = _dibujar_cajas(img, cajas, (0, 255, 255), 1)
            imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), color_img=True,
                   title=f"6.{idx + 1} - Candidatos tipo carácter ({len(cajas)})", blocking=True)
        for run, sc in _evaluar_filas(cajas):
            ch = float(np.median([b[3] for b in run]))
            sc += 8.0 * _blancura_fondo(hsv, run) # fondo blanco de la chapa
            if _hay_azul_arriba(hsv, run, ch):
                sc += 2.5 # vemos la franja azul para estar mas seguros
            if sc > mejor_score:
                mejor_score, mejor_fila = sc, run

    if debug and mejor_fila:
        vis = _dibujar_cajas(img, mejor_fila, (0, 255, 0), 2)
        imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), color_img=True,
               title=f"7 - Mejor fila encontrada (score={mejor_score:.2f})", blocking=True)
    return mejor_fila


def detectar_patente(img_bgr, debug=False):
    """Detecta la patente buscando directamente la FILA DE 7 CARACTERES que
    cumple la geometría oficial de la patente Mercosur. Devolvemos la patente y el bbox."""
    H0, W0 = img_bgr.shape[:2]
    target_w = 1100 # aumentamos la resolucion para patentes lejanas
    scale = target_w / float(W0)
    img = cv2.resize(img_bgr, (target_w, int(H0 * scale)),
                     interpolation=cv2.INTER_AREA)

    if debug:
        imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), color_img=True,
               title="1 - Imagen redimensionada", blocking=True)

    fila = _buscar_fila_patente(img, debug=debug)
    if not fila:
        if debug:
            print("No se encontró ninguna fila de caracteres válida.")
        return None, None

    cx0 = min(b[0] for b in fila)
    cy0 = min(b[1] for b in fila)
    cx1 = max(b[0] + b[2] for b in fila)
    cy1 = max(b[1] + b[3] for b in fila)
    ch = float(np.median([b[3] for b in fila]))
    row_w = cx1 - cx0

    mx = int(0.12 * row_w + ch)
    my_top = int(0.90 * ch)                 # franja azul arriba
    my_bot = int(0.55 * ch)
    bx0 = max(0, cx0 - mx); by0 = max(0, cy0 - my_top)
    bx1 = min(img.shape[1], cx1 + mx); by1 = min(img.shape[0], cy1 + my_bot)
    crop = img[by0:by1, bx0:bx1]

    if debug and crop.size:
        imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), color_img=True,
               title="8 - Región recortada para refinar", blocking=True)

    # Comparamos con la geometría de la patente para mas precision
    cajas, _ = _detectar_chars(crop)
    if debug and len(cajas):
        vis = _dibujar_cajas(crop, cajas, (255, 0, 0), 1)
        imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), color_img=True,
               title=f"9 - Caracteres refinados sobre el crop ({len(cajas)})", blocking=True)

    if len(cajas) >= 4:
        dx0 = min(c[0] for c in cajas); dy0 = min(c[1] for c in cajas)
        dx1 = max(c[0] + c[2] for c in cajas); dy1 = max(c[1] + c[3] for c in cajas)
        chh = dy1 - dy0
        px = max(int(0.06 * (dx1 - dx0)), int(0.3 * chh))
        py_top = int(0.75 * chh)
        py_bot = int(0.40 * chh)
        x = bx0 + max(0, dx0 - px); y = by0 + max(0, dy0 - py_top)
        x1 = bx0 + dx1 + px; y1 = by0 + dy1 + py_bot
    else:
        px = int(0.05 * row_w)
        x = max(0, cx0 - px); y = max(0, cy0 - int(0.70 * ch))
        x1 = min(img.shape[1], cx1 + px); y1 = min(img.shape[0], cy1 + int(0.40 * ch))

    if debug:
        vis = img.copy()
        for b in fila:
            cv2.rectangle(vis, (b[0], b[1]),
                          (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 1)
        cv2.rectangle(vis, (x, y), (x1, y1), (0, 0, 255), 2)
        imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), color_img=True,
               title="10 - BBox final de la patente", blocking=True)

    x0o = int(x / scale); y0o = int(y / scale)
    x1o = int(x1 / scale); y1o = int(y1 / scale)
    patente = img_bgr[y0o:y1o, x0o:x1o].copy()
    return patente, (x0o, y0o, x1o - x0o, y1o - y0o)


if __name__ == "__main__":
    # Uso: python detectar_patente.py <ruta_imagen>
    ruta = sys.argv[1] if len(sys.argv) > 1 else "patente.jpg"

    if not os.path.exists(ruta):
        print(f"No existe el archivo: {ruta}")
        print("Uso: python detectar_patente.py <ruta_imagen>")
        sys.exit(1)

    img = cv2.imread(ruta)
    if img is None:
        print(f"No se pudo leer la imagen: {ruta}")
        sys.exit(1)

    patente, bbox = detectar_patente(img, debug=True)

    if patente is None:
        print("No se detectó ninguna patente.")
    else:
        print(f"Patente detectada. BBox (x, y, w, h): {bbox}")
        imshow(cv2.cvtColor(patente, cv2.COLOR_BGR2RGB), color_img=True,
               title="Resultado final - Patente recortada", blocking=True)