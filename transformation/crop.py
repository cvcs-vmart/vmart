import cv2
from utils.img import show_img, save_img
import numpy as np
import math


def crop(img, x1, y1, x2, y2, pad=10):
    x1 = max(x1 - pad, 0)
    x2 = min(x2 + pad, img.shape[1])

    y1 = max(y1 - pad, 0)
    y2 = min(y2 + pad, img.shape[0])

    # save_img(img[y1:y2, x1:x2], f'data/img/cropped-{x1}-{y1}-{x2}-{y2}.jpg')

    return img[y1:y2, x1:x2]


def resize_img(img, TARGET=512):
    height, width = img.shape[:2]

    TARGET = 512
    lmax = max(height, width)
    lmin = min(height, width)

    ratio = float(lmin) / float(lmax)

    l_max = TARGET
    l_min = int(TARGET * ratio)

    if max == height:
        return cv2.resize(img, (l_max, l_min), interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(img, (l_min, l_max), interpolation=cv2.INTER_LINEAR)


