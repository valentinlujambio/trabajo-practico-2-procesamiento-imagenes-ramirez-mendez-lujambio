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
    
    patente = detectar_patente(img_bgr, debug=False)
    if patente is None:
        print(f"  [!] No se detectó patente en {os.path.basename(path)}")
        if debug:
            imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), color_img=True, title=f"{os.path.basename(path)} - sin detección")
        return


if __name__ == "__main__":
    paths = sorted(glob.glob('img/img_*.jpg'))
    print(f"Procesando {len(paths)} imágenes...")
    for p in paths:
        print(f"\n>>> {os.path.basename(p)}")
        procesar_imagen(p, debug=False)
    plt.show()
