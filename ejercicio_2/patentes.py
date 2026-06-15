"""
Ejercicio 2 - Detección de patentes y segmentación de caracteres
TP2 - PDI TUIA 2026 C1
"""
import glob
import os
import cv2
import matplotlib.pyplot as plt


from help_show import imshow
from placa import detectar_patente
from caracteres import segmentar_caracteres

# listado_patentes = [
#     "img_1":"AG678UA",
#     "img_2":"AB000RT",
#     "img_3":"AC164JM",
#     "img_4":"AG456NR",
#     "img_5":"AH482KU",
#     "img_6":"AG000GA",
#     "img_7":"AE001ET",
#     "img_8":"AH486ML",
#     "img_9":"AI003UM",
#     "img_10":"AE233UG",
#     "img_11":"AA017EA",
#     "img_12":"AC712QM",
# ]


def procesar_imagen(path, debug=False):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"  [!] No se pudo leer {path}")
        return
    
    patente, bbox = detectar_patente(img_bgr, debug=False)
    if patente is None:
        print(f"  [!] No se detectó patente en {os.path.basename(path)}")
        if debug:
            imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), color_img=True, title=f"{os.path.basename(path)} - sin detección")
        return
    chars, patente_resized = segmentar_caracteres(patente, debug=False)
    print(f"  patente detectada en bbox {bbox} - {len(chars)} caracteres")
    
    if debug:
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(os.path.basename(path))

        vis = img_bgr.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Original con patente detectada")
        plt.xticks([]); plt.yticks([])

        vis2 = patente_resized.copy()
        scale = patente_resized.shape[0] / float(patente.shape[0])
        for (cx, cy, cw, ch) in chars:
            rx = int(cx * scale); ry = int(cy * scale)
            rw = int(cw * scale); rh = int(ch * scale)
            cv2.rectangle(vis2, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
        plt.title(f"patente - {len(chars)} caracteres segmentados")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    paths = sorted(glob.glob('img/img_*.jpg'))
    print(f"Procesando {len(paths)} imágenes...")
    for p in paths:
        print(f"\n>>> {os.path.basename(p)}")
        procesar_imagen(p, debug=False)
    plt.show()
