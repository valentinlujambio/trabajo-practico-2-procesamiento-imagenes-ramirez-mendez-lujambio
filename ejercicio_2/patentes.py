"""
Ejercicio 2 - Detección de placas patente y segmentación de caracteres
TP2 - PDI TUIA 2026 C1
"""
import glob
import os
import cv2
import matplotlib.pyplot as plt

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


def procesar_imagen(path, debug=True):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"  [!] No se pudo leer {path}")
        return


if __name__ == "__main__":
    paths = sorted(glob.glob('img/img_*.jpg'))
    print(f"Procesando {len(paths)} imágenes...")
    for p in paths:
        print(f"\n>>> {os.path.basename(p)}")
        procesar_imagen(p, debug=True)
    plt.show()
