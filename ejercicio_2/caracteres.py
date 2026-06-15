import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from help_show import imshow

def _extraer_cajas_chars(th):
    """Extraemos las cajas de los caracteres de la imagen"""
    num, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    Hp, Wp = th.shape
    cajas = []
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if ch < 12 or ch > 0.95 * Hp:
            continue
        ar = cw / float(ch)
        if ar < 0.10 or ar > 1.10:
            continue
        cajas.append((x, y, cw, ch))
    return cajas


def _linea_dominante(cajas):
    """Buscamos la linea de caracteres dominante"""
    if not cajas:
        return []
    mejor = []
    for seed in cajas:
        sh = seed[3]
        scy = seed[1] + seed[3] / 2.0
        grupo = []
        for b in cajas:
            bh = b[3]
            bcy = b[1] + b[3] / 2.0
            if 0.65 * sh <= bh <= 1.5 * sh and abs(bcy - scy) <= 0.6 * sh:
                grupo.append(b)
        # desempate: más miembros y mayor cobertura horizontal
        if len(grupo) > len(mejor):
            mejor = grupo
    return sorted(mejor, key=lambda b: b[0])


def _detectar_chars(placa_bgr, debug=False):
    """Detectamos los caracteres en un recorte de placa"""
    h, w = placa_bgr.shape[:2]
    if h == 0 or w == 0:
        return [], placa_bgr
    h_target = 130
    scale = h_target / float(h)
    placa = cv2.resize(placa_bgr, (max(1, int(w * scale)), h_target),
                        interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    # Realizamos un CLAHE para mejorar el contraste de la imagen
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Probamos varias configuraciones de adaptiveThreshold + Otsu (fallback)
    configs = [('gauss', 21, 8), ('gauss', 31, 10), ('gauss', 15, 6), ('mean',  21, 8), ('otsu',  0,  0)]
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    mejor_cajas, mejor_th, mejor_score = [], None, -float('inf')
    for tipo, bs, C in configs:
        if tipo == 'gauss':
            th = cv2.adaptiveThreshold(
                gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, bs, C)
        elif tipo == 'mean':
            th = cv2.adaptiveThreshold(
                gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, bs, C)
        else:
            _, th = cv2.threshold(gray_eq, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
        cajas = _linea_dominante(_extraer_cajas_chars(th))
        score = -abs(len(cajas) - 7)
        if score > mejor_score:
            mejor_score, mejor_cajas, mejor_th = score, cajas, th
            if score == 0:
                break

    cajas = mejor_cajas
    if len(cajas) > 7:
        h_med = float(np.median([b[3] for b in cajas]))
        cajas = sorted(cajas, key=lambda b: abs(b[3] - h_med))[:7]
    cajas.sort(key=lambda b: b[0])

    if debug and mejor_th is not None:
        imshow(mejor_th, title="B.1 - mejor binarización")

    cajas_orig = [(int(x / scale), int(y / scale),
                    int(cw / scale), int(ch / scale))
                    for (x, y, cw, ch) in cajas]
    return cajas_orig, placa


def segmentar_caracteres(placa_bgr, debug=False):
    cajas_orig, placa = _detectar_chars(placa_bgr, debug=debug)
    if debug:
        scale = placa.shape[0] / float(placa_bgr.shape[0])
        vis = placa.copy()
        for (x, y, w_, h_) in cajas_orig:
            cv2.rectangle(vis, (int(x*scale), int(y*scale)),
                            (int((x+w_)*scale), int((y+h_)*scale)), (0, 255, 0), 2)
        imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                color_img=True, title=f"B.2 - {len(cajas_orig)} caracteres")
    return cajas_orig, placa

