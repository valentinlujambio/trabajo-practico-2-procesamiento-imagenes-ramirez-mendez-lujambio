import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False,
           colorbar=False, ticks=False):
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
