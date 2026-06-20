"""
Ejercicio 1 - Detección y clasificación de pastillas
TP2 - PDI TUIA 2026 C1

Versión con MODO DEBUG: poné DEBUG = True en el __main__ para ver
todas las salidas intermedias de cada etapa (A, B, C).
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


def _panel(items, suptitle):
    """
    [DEBUG] Muestra varias imágenes intermedias en una sola figura.
    items: lista de tuplas (imagen, titulo, es_color)
    """
    n = len(items)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (im, ttl, es_color) in zip(axes, items):
        if es_color:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap='gray')
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:           # apagar ejes sobrantes
        ax.axis('off')
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show(block=False)


def segmentar_cinta(img_gray, debug=False):
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

    if debug:
        comp = np.uint8(labels == idx) * 255  # componente elegido aislado
        print(f"[A] segmentar_cinta: {num - 1} componentes detectados | "
              f"elegido idx={idx} area={int(areas.max())} "
              f"bbox=({x},{y},{w},{h}) pad={pad}")
        _panel([
            (img_gray, "1. Entrada (gray)", False),
            (th,       "2. Otsu invertido (cinta blanca)", False),
            (opened,   "3. Apertura RECT 80x30", False),
            (closed,   "4. Cierre RECT 55x55", False),
            (comp,     "5. Componente mayor = cinta", False),
            (mask,     "6. Mascara final (con pad)", False),
        ], "A - segmentar_cinta")

    return mask, (x0, y0, x1 - x0, y1 - y0)


def detectar_pastillas(img_gray, mask_cinta, area_min=1500, min_dim=25,
                       img_hsv=None, debug=False):
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
    th_belt = cv2.bitwise_and(th, mask_cinta)

    # Aplicamos opening y closing para limpiar el ruido
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(th_belt, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # Nos quedamos con los componentes conectados
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8)

    pastillas = []
    descartados = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < area_min:
            descartados.append((i, int(area), w, h, "area<area_min"))
            continue
        # Descartamos tiras finas (artefactos del borde de la cinta que nos dan problemas)
        if min(w, h) < min_dim:
            descartados.append((i, int(area), w, h, "min(w,h)<min_dim"))
            continue
        mask = np.uint8(labels == i) * 255
        pastillas.append({
            'bbox': (x, y, w, h),
            'mask': mask,
            'area': int(area),
            'centroid': (float(centroids[i, 0]), float(centroids[i, 1])),
        })

    if debug:
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        for p in pastillas:
            x, y, w, h = p['bbox']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"[B] detectar_pastillas: {num - 1} componentes | "
              f"aceptados={len(pastillas)} descartados={len(descartados)}")
        for d in descartados:
            print(f"      descartado idx={d[0]} area={d[1]} "
                  f"wh=({d[2]},{d[3]}) motivo={d[4]}")
        _panel([
            (V,        "1. Canal V (brillo)", False),
            (S,        "2. Canal S (saturacion)", False),
            (th_v,     "3. Otsu(V)", False),
            (th_s,     "4. Otsu(S)", False),
            (th,       "5. V OR S", False),
            (th_belt,  "6. AND con mascara cinta", False),
            (opened,   "7. Apertura ELLIPSE 3x3", False),
            (cleaned,  "8. Cierre ELLIPSE 11x11", False),
            (overlay,  "9. Detecciones aceptadas", True),
        ], "B - detectar_pastillas")

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


def clasificar_pastilla(mask, img_hsv, debug=False, etq=""):
    """
    Punto C: Clasifica la pastilla segun sus caracteristicas.
    """
    # Tenemos dos caracteristicas que nos interesan: forma y color, sino obtenemos una forma conocida devolvemos XX
    f = _features_forma(mask)
    c = _features_color(img_hsv, mask)
    if f is None:
        if debug:
            print(f"  [{etq}] sin contorno -> XX")
        return 'XX'

    # En base a la forma, después vemos el color
    es_capsula = (f['aspect'] >= 1.7) or (f['circularity'] < 0.65)
    if es_capsula:
        if c['pct_blue'] > 0.05:
            sigla = 'CB'    # Capsula bicolor azul-blanca
        elif c['pct_yellow'] > 0.20:
            sigla = 'CA'    # Capsula amarilla
        else:
            sigla = 'CB'
    elif f['circularity'] < 0.88:
        sigla = 'CC'        # Cuadrada blanca
    elif c['pct_pink'] > 0.15:
        sigla = 'RR'        # Redonda rosa
    else:
        sigla = 'RB'        # Redonda blanca

    if debug:
        print(f"  [{etq}] circ={f['circularity']:.3f} aspect={f['aspect']:.2f} "
              f"extent={f['extent']:.3f} | "
              f"blue={c['pct_blue']:.2f} yellow={c['pct_yellow']:.2f} "
              f"pink={c['pct_pink']:.2f} white={c['pct_white']:.2f} "
              f"| capsula={es_capsula} -> {sigla}")
    return sigla


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
    DEBUG = True   # <-- poné False para volver al modo normal

    img_bgr = cv2.imread('img/pills.png')
    if img_bgr is None:
        raise FileNotFoundError("No se encontró 'img/pills.png'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # Para distinguir cápsulas bicolor

    if DEBUG:
        imshow(img_rgb, color_img=True, title="Imagen original")

    # A - segmentación de la cinta
    mask_cinta, bbox_cinta = segmentar_cinta(img_gray, debug=DEBUG)

    # B - detección de pastillas
    pastillas, cleaned = detectar_pastillas(
        img_gray, mask_cinta, img_hsv=img_hsv, debug=DEBUG)

    # C - clasificación
    if DEBUG:
        print("\n=== Features por pastilla (orden de detección) ===")
    etiquetas = []
    for k, p in enumerate(pastillas, start=1):
        etiquetas.append(
            clasificar_pastilla(p['mask'], img_hsv, debug=DEBUG, etq=f"#{k}"))

    # D - reporte
    generar_salida(img_rgb, pastillas, etiquetas)

    plt.show()   # bloquea al final para que queden abiertas todas las figuras