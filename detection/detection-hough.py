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

from collections import defaultdict


DEBUG = True

IMG_NAME = 'art-gallery'
IMG_SAVE_DIR = f'data/img/{IMG_NAME}/'

img = cv2.imread(f'data/img/{IMG_NAME}.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width = img.shape[:2]
heightScale = 768 / height
widthScale = 1024 / width

img = cv2.resize(img, (int(widthScale * width), int(heightScale * height)), interpolation=cv2.INTER_LINEAR)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

with timer.Timer('Canny'):
    # Gaussian blur
    img_mod = cv2.GaussianBlur(img, (5, 5), 2)

    show_img(img_mod, 'Gaussian Blur')

    # Canny edge detection
    edges = cv2.Canny(img_mod, 85, 239, L2gradient=True)

    show_img(edges, 'Canny Edges')

    # Type casting
    edges = (edges > 0).astype(np.uint8) * 255

    # Debug mode
    if DEBUG:
        show_img(edges, 'Canny Edges')
        save_img(edges, f'data/img/canny-{IMG_NAME}.jpg')

    # Corner detection
    # corner = cv2.cornerHarris(img_gray, blockSize=3, ksize=3, k=0.03)
    # corner = cv2.dilate(corner, None, iterations=1)

    # Hough Transform (probabilistic)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=45,
                            minLineLength=55,
                            maxLineGap=20)

    # Draw lines
    horizontal_lines = []
    vertical_lines = []
    if lines is not None:
        img_mod_hough = img_mod.copy()
        img_mod_hough = cv2.cvtColor(img_mod_hough, cv2.COLOR_BGR2RGB)
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            # cv2.line(edges_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if oblique(x1, y1, x2, y2, threshold=90):
                pass
            else:
                ex1, ey1, ex2, ey2, R, G, B = extend_line_partial(x1, y1, x2, y2, edges.shape[1], edges.shape[0])

                cv2.line(img_mod_hough, (ex1, ey1), (ex2, ey2), color=(R, G, B), thickness=2)
                if (R, G, B) == (255, 0, 0):
                    horizontal_lines.append((ex1, ey1, ex2, ey2))
                elif (R, G, B) == (0, 255, 0):
                    vertical_lines.append((ex1, ey1, ex2, ey2))

    if DEBUG:
        show_img(img_mod_hough, 'Hough Probabilistic - Grouped Lines')
        #save_img(img_mod_hough, f'data/img/hough-{IMG_NAME}.jpg')

    # Interception points
    interception_points = []  # this contains the points of interception between horizontal and vertical lines and the lines
    lines_points = defaultdict(list)
    just_points = []
    if horizontal_lines is not None and vertical_lines is not None:
        for (hx1, hy1, hx2, hy2) in horizontal_lines:
            for (vx1, vy1, vx2, vy2) in vertical_lines:
                if intersect((hx1, hy1), (hx2, hy2), (vx1, vy1), (vx2, vy2)):
                    # Calculate the intersection point
                    ix, iy = intersection_point((hx1, hy1), (hx2, hy2), (vx1, vy1), (vx2, vy2))

                    just_points.append((ix, iy))

                    chx = (hx1 + hx2) / 2
                    chy = (hy1 + hy2) / 2

                    cvx = (vx1 + vx2) / 2
                    cvy = (vy1 + vy2) / 2

                    interception_points.append(
                        ((ix, iy), (hx1, hy1, hx2, hy2, chx, chy), (vx1, vy1, vx2, vy2, cvx, cvy)))
                    cv2.circle(img_mod_hough, (ix, iy), 3, (0, 0, 255), -1)

                    # Store the points of the lines
                    lines_points[(hx1, hy1, hx2, hy2)].append((ix, iy))
                    lines_points[(vx1, vy1, vx2, vy2)].append((ix, iy))

    if DEBUG:
        show_img(img_mod_hough, 'Hough Probabilistic - Interception Points')
        #save_img(img_mod_hough, f'data/img/hough-interception-{IMG_NAME}.jpg')

    suppressed_points = keypoint_suppression(just_points, 16)

    # Draw the clamped lines
    img_mod_suppression_keypoint = img_mod.copy()
    img_mod_suppression_keypoint = cv2.cvtColor(img_mod_suppression_keypoint, cv2.COLOR_BGR2RGB)
    keypoints_coords = np.argwhere(suppressed_points == 1)
    for y, x in keypoints_coords:
        cv2.circle(img_mod_suppression_keypoint, (x, y), 5, (0, 255, 0), -1)


    if DEBUG:
        show_img(img_mod_suppression_keypoint, 'Hough Probabilistic - Interception Lines Clamped')
        #save_img(img_mod_clamp_hough, f'data/img/hough-interception-clamped-{IMG_NAME}.jpg')