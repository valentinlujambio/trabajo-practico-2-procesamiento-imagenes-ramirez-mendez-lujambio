"""
Ejercicio 1 - Detección y clasificación de pastillas
TP2 - PDI TUIA 2026 C1
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# =============================================================================
# ETAPA A - Segmentación del ROI de la cinta transportadora
# =============================================================================
def segmentar_cinta(img_gray):
    """
    Devuelve una máscara del área de la cinta.
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



# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    img_bgr = cv2.imread('img/pills.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    imshow(img_rgb, color_img=True, title="Imagen original")

    # A
    mask_cinta, bbox_cinta = segmentar_cinta(img_gray)
    imshow(mask_cinta, title="A - Máscara de la cinta")
    
    plt.show()
