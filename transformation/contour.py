import random

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

def order_points(pts):
    # ordina in base alla somma: top-left avrà somma minima, bottom-right massima
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)  # x - y

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


### Parametri Gaussian blur
kernel_size = 3
sigma = 2

### Parametri canny
canny_threshold1 = 85
canny_threshold2 = 239
L2gradient = True

### Parametri filtraggio contorni
CONTOUR_AREA_THRESHOLD = 100

### Parametri approssimazione contorni
approx_epsilon = 0.05

DEBUG = True
SAVE_TRANSFORMATION = True


def find_contours(img, id):
    h, w, c = img.shape

    show_img(img, 'Original Image')
    save_img(img, f'data/img/original-{id}.jpg')
    # immagine in grigio per canny
    #img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    img_blur = cv2.bilateralFilter(img, 3, 75, 150)

    edges = cv2.Canny(img_blur, canny_threshold1, canny_threshold2, L2gradient=L2gradient)

    if DEBUG:
        show_img(edges, 'Canny Edges')
        save_img(edges, f'data/img/canny-{id}.jpg')

    # Chiusura morfologica per unire il contorno esterno
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    #edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if DEBUG:
        show_img(edges_closed, 'Closed Edges')
        save_img(edges_closed, f'data/img/closed-edges-{id}.jpg')

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        img_contours = edges_closed.copy()
        img_contours[img_contours > 0] = 0
        for contour in contours:
            cv2.drawContours(img_contours, [contour], -1, (255, 255, 255), 2)
        show_img(img_contours, 'Contours')
        save_img(img_contours, f'data/img/contours-{id}.jpg')

    # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD]

    # if DEBUG:
    #     img_contours = edges_closed.copy()
    #     img_contours[img_contours > 0] = 0
    #     for contour in contours:
    #         cv2.drawContours(img_contours, [contour], -1, (255, 255, 255), 2)
    #     show_img(img_contours, 'Filtered Contours')
    #     save_img(img_contours, f'data/img/filtered-contours-{id}.jpg')

    approx_contours = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, approx_epsilon * peri, closed=True)
        if len(approx) == 4:
            approx_contours.append(approx)

    if DEBUG:
        img_approx = img.copy()
        for approx in approx_contours:
            cv2.drawContours(img_approx, [approx], -1, (0, 0, 255), 2)
        show_img(img_approx, 'Approximated Contours')
        save_img(img_approx, f'data/img/approximated-contours-{id}.jpg')

    approx_contours = sorted(approx_contours, key=lambda x: cv2.contourArea(x), reverse=True)

    winner = approx_contours[0] if approx_contours else None
    if DEBUG:
        img_approx_sorted = img.copy()
        for approx in approx_contours:
            winner = approx
            cv2.drawContours(img_approx_sorted, [approx], -1, (0, 0, 255), 2)
            break
        show_img(img_approx_sorted, 'Sorted Approximated Contours')
        save_img(img_approx_sorted, f'data/img/sorted-approximated-contours-{id}.jpg')


    target_area = h * w
    if winner is not None:
        if cv2.contourArea(winner) > target_area * 0.65:
            winner = np.squeeze(winner)
            winner = order_points(winner)

            p1, p2, p3, p4 = winner


            winner = np.array([p1, p2, p3, p4], dtype=np.float32)

            target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

            out = np.zeros((h, w, c))

            #M, mask = cv2.findHomography(winner, target)
            M, mask = cv2.findHomography(winner, target, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is not None:
                out = cv2.warpPerspective(img, M, (out.shape[1], out.shape[0]))

            if DEBUG:
                show_img(out, 'Warped Image')

            if SAVE_TRANSFORMATION:
                save_img(out, f'data/img/transformed-{id}.jpg')

            print('Transformation completed successfully')
            return out

    print('No valid contour found or contour area is too small')
    if SAVE_TRANSFORMATION:
        save_img(img, f'data/img/not-transformed-{id}.jpg')
    return img



def hough_contours(img):
    pass
