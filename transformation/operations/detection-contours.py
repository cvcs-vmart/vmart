import torch
import cv2
import numpy as np
from itertools import product
from skimage.morphology import thin, skeletonize
from matplotlib import pyplot as plt

from utils import timer
from utils.geometry import (oblique, extend_line, extend_line_partial, intersect,
                            intersection_point, ccw, keypoint_suppression, get_rectangular_contours)
from utils.img import show_img, save_img


DEBUG = True

IMG_NAME = 'art-gallery1'
IMG_SAVE_DIR = f'data/img/{IMG_NAME}/'

img = cv2.imread(f'data/img/{IMG_NAME}.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width = img.shape[:2]
heightScale = 768 / height
widthScale = 1024 / width

img = cv2.resize(img, (int(widthScale * width), int(heightScale * height)), interpolation=cv2.INTER_LINEAR)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# (kernel size, sigma)
GAUSSIAN_PARAMS = [(3, 1)]

# Area Threshold for contours
CONTOUR_AREA_THRESHOLD = 3000

# Saving rectangles and their scores
rects = [None] * len(GAUSSIAN_PARAMS)

with timer.Timer('Canny'):
    for i, (kernel_size_gaussian_blur, sigma_gaussian_blur) in enumerate(GAUSSIAN_PARAMS):

        #img_mod = LAB(img)
        # Gaussian Blur
        # img_mod = cv2.GaussianBlur(img, (kernel_size_gaussian_blur, kernel_size_gaussian_blur), sigma_gaussian_blur)
        img_mod = cv2.bilateralFilter(img, 5, 75, 150)

        # Canny edge detection
        edges = cv2.Canny(img_mod, 85, 239)

        # Closing operation
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Find contours (on edges)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD]

        edges_copy = edges_closed.copy()
        edges_copy[edges_copy > 0] = 0
        for contour in contours:
            cv2.drawContours(edges_copy, [contour], -1, (255, 255, 255), 2)


        new_img = img.copy()
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:
                cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        show_img(new_img, f'Bounding Rect - Gaussian Blur Kernel Size: {kernel_size_gaussian_blur}, Sigma: {sigma_gaussian_blur}')

        r, s = get_rectangular_contours(contours)
        rects[i] = (r, s)

        img_rect = img.copy()
        img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
        for r in rects[i][0]:
            cv2.drawContours(img_rect, [r], -1, (0, 0, 255), 2)

        if DEBUG:
            show_img(img_rect, f'Rectangular Contours - Gaussian Blur Kernel Size: {kernel_size_gaussian_blur}, Sigma: {sigma_gaussian_blur}')
            #save_img(img_rect, f'data/img/rect-{IMG_NAME}.jpg')

        if DEBUG:
            show_img(edges_copy, 'Contours')
            #save_img(edges_copy, f'data/img/contours-{IMG_NAME}.jpg')

    # Save the image with the best rectangles
    rects = sorted(rects, key=lambda x: (-len(x[0]), x[1]))

    best_rect, _ = rects[0]

    img_rect = img.copy()
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    for r in best_rect:
        cv2.drawContours(img_rect, [r], -1, (0, 0, 255), 2)

    if DEBUG:
        show_img(img_rect, 'Best Rectangular Contours')
        #save_img(img_rect, f'data/img/best-rect-{IMG_NAME}.jpg')

