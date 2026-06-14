"""
Ejercicio 1 - Detección y clasificación de pastillas
TP2 - PDI TUIA 2026 C1
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False,
           colorbar=False, ticks=False):
  """
  Ayuda de visualización
  """
  if new_fig:
      plt.figure()
  if color_img:
      plt.imshow(img)
  else:
      plt.imshow(img, cmap='gray')
  plt.title(title)
  if not ticks:
      plt.xticks([]), plt.yticks([])
  if colorbar:
      plt.colorbar()
  if new_fig:
      plt.show(block=blocking)


def segmentar_cinta(img_gray):
    """
    Punto A: Devuelve una máscara del área de la cinta.
    """
    # Umbralizamos la cinta, dejandola blanca
    _, th = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 30))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)

    # Rellenamos los huecos que dejan las pastillas.
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # Nos quedamos con el componente mas grande, que es la cinta
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num <= 1:
        raise RuntimeError("No se detectó la cinta")
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]

    # Devolvemos la mascara rectangular, con un shrink hacia adentro
    # para evitar tocar los frames metálicos del borde, porque varias veces tuvimos problemas.
    pad = max(5, int(0.02 * min(w, h)))
    x0 = x + pad; y0 = y + pad
    x1 = x + w - pad; y1 = y + h - pad
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1 - x0, y1 - y0)


def detectar_pastillas(img_gray, mask_cinta, area_min=1500, min_dim=25, img_hsv=None):
    """
    Punto B: Detecta pastillas combinando brillo (V) y saturación (S). Devolvemos una lista de pastillas con sus caracteristicas.
    """
    if img_hsv is None:
        raise ValueError("Se necesita img_hsv para distinguir cápsulas bicolor")

    V = img_hsv[:, :, 2]
    S = img_hsv[:, :, 1]

    # Umbralizamos V (claros) y S (saturados) con Otsu independiente
    _, th_v = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_or(th_v, th_s)

    # Restringir a la cinta
    th = cv2.bitwise_and(th, mask_cinta)

    # Aplicamos opening y closing para limpiar el ruido
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # Nos quedamos con los componentes conectados
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)

    pastillas = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # Descartamos tiras finas (artefactos del borde de la cinta que nos dan problemas)
        if min(w, h) < min_dim:
            continue
        mask = np.uint8(labels == i) * 255
        pastillas.append({
            'bbox': (x, y, w, h),
            'mask': mask,
            'area': int(area),
            'centroid': (float(centroids[i, 0]), float(centroids[i, 1])),
        })
    return pastillas, cleaned


def _features_forma(mask):
    """
    Ayuda para calcular las caracteristicas de la forma de la pastilla.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (per * per + 1e-6)

    (cx, cy), (mw, mh), ang = cv2.minAreaRect(cnt)
    if mw < 1 or mh < 1:
        return None
    aspect = max(mw, mh) / min(mw, mh)

    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h + 1e-6)

    return {'circularity': circularity, 'aspect': aspect, 'extent': extent}


def _features_color(img_hsv, mask):
    """
    Ayuda para calcular las caracteristicas del color de la pastilla.
    """
    mean = cv2.mean(img_hsv, mask=mask)  # (H, S, V, 0)
    h_mean, s_mean, v_mean = mean[0], mean[1], mean[2]

    # Obtenemos los porcentajes de píxeles por color dentro del blob
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]
    m = mask > 0

    # Definimos los rangos del color
    blue   = ((H >= 90)  & (H <= 135) & (S >= 60)) & m
    yellow = ((H >= 15)  & (H <= 40)  & (S >= 50)) & m
    pink   = (((H >= 150) | (H <= 10))& (S >= 30)) & m
    white  = ((S < 50) & (V > 130)) & m

    n = max(1, m.sum())
    return {
        'H': h_mean, 'S': s_mean, 'V': v_mean,
        'pct_blue':   blue.sum()   / n,
        'pct_yellow': yellow.sum() / n,
        'pct_pink':   pink.sum()   / n,
        'pct_white':  white.sum()  / n,
    }


def clasificar_pastilla(mask, img_hsv):
    """
    Punto C: Clasifica la pastilla segun sus caracteristicas.
    """
    # Tenemos dos caracteristicas que nos interesan: forma y color, sino obtenemos una forma conocida devolvemos XX
    f = _features_forma(mask)
    c = _features_color(img_hsv, mask)
    if f is None:
        return 'XX'

    # En base a la forma, después vemos el color
    es_capsula = (f['aspect'] >= 1.7) or (f['circularity'] < 0.65)
    if es_capsula:
        if c['pct_blue'] > 0.05:
            return 'CB' # Capsula bicolor azul-blanca
        if c['pct_yellow'] > 0.20:
            return 'CA' # Capsula amarilla
        return 'CB'

    if f['circularity'] < 0.88:
        return 'CC' # Cuadrada blanca

    if c['pct_pink'] > 0.15:
        return 'RR' # Redonda rosa
    return 'RB' # Redonda blanca


NOMBRES = {
    'RB': 'Redonda blanca',
    'RR': 'Redonda rosa',
    'CC': 'Cuadrada blanca',
    'CA': 'Capsula amarilla',
    'CB': 'Capsula azul-blanca',
    'XX': 'Desconocida',
}


def generar_salida(img_rgb, pastillas, etiquetas, out_path='salida_pastillas.png'):
    cuenta = Counter(etiquetas)
    print("\n=== Resultado de la clasificación ===")
    total = 0
    for sigla, n in sorted(cuenta.items()):
        nombre = NOMBRES.get(sigla, sigla)
        print(f"  {sigla} ({nombre:<22}): {n}")
        total += n
    print(f"  {'TOTAL':<28}: {total}")

    # Dibujar etiquetas con id por tipo (RR1, RR2, ...)
    contadores = {}
    out = img_rgb.copy()
    for p, sigla in zip(pastillas, etiquetas):
        contadores[sigla] = contadores.get(sigla, 0) + 1
        etq = f"{sigla}{contadores[sigla]}"
        x, y, w, h = p['bbox']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, etq, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    imshow(out, color_img=True, title="Pastillas clasificadas")
    print(f"\nImagen guardada en: {out_path}")
    return out


if __name__ == "__main__":
    img_bgr = cv2.imread('img/pills.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) # Para distinguir cápsulas bicolor
    
    imshow(img_rgb, color_img=True, title="Imagen original")

    # A
    mask_cinta, bbox_cinta = segmentar_cinta(img_gray)
    imshow(mask_cinta, title="A - Máscara de la cinta")
    
    # B
    pastillas = detectar_pastillas(img_gray, mask_cinta)
    imshow(pastillas, title="B - Pastillas")
    
    # C
    etiquetas = [clasificar_pastilla(p['mask'], img_hsv) for p in pastillas]
    imshow(etiquetas, title="C - Etiquetas")
    
    # D - reporte
    generar_salida(img_rgb, pastillas, etiquetas)
    
    plt.show()
