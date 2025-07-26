import math
import numpy as np
import cv2

def single_angle_evaluation(rect, threshold=10):
    p1 = rect[0][0][:]
    p2 = rect[1][0][:]
    p3 = rect[2][0][:]
    p4 = rect[3][0][:]

    if abs(90 - angle_between(p1, p2, p3)) > threshold:
        return False
    if abs(90 - angle_between(p2, p3, p4)) > threshold:
        return False
    if abs(90 - angle_between(p3, p4, p1)) > threshold:
        return False
    if abs(90 - angle_between(p4, p1, p2)) > threshold:
        return False

    return True

def angle_between(p1, p2, p3):

    # Obtaining the two vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Dot product and norm
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(np.clip(dot / norm_product, -1.0, 1.0))

    return np.degrees(angle_rad)

def evaluate_rect_angles(rect):
    p1 = rect[0][0][:]
    p2 = rect[1][0][:]
    p3 = rect[2][0][:]
    p4 = rect[3][0][:]

    # Calculate the angles between the sides
    angle1 = angle_between(p4, p1, p2)
    angle2 = angle_between(p1, p2, p3)
    angle3 = angle_between(p2, p3, p4)
    angle4 = angle_between(p3, p4, p1)

    return abs(90 - angle1) + abs(90 - angle2) + abs(90 - angle3) + abs(90 - angle4)




def get_rectangular_contours(contours):
    """Approximates provided contours and returns only those which have 4 vertices"""
    res = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, 0.05 * peri, closed=True)
        if len(approx) == 4:
            res.append(approx)

    # output = []
    # score_tot = 0
    # for contour in res:
    #     score = evaluate_rect_angles(contour)
    #     score_tot += score
    #     if score < 40:
    #         output.append(contour)
    #     else:

    output = []
    score_tot = 0
    for contour in res:
        if single_angle_evaluation(contour):
            output.append(contour)
            score_tot += evaluate_rect_angles(contour)

    return output, score_tot

def keypoint_suppression(points, window_size=8):

    img_points = np.zeros((768, 1028), dtype=np.uint8)
    for point in points:
        x, y = point
        img_points[y, x] = 1


    h, w = img_points.shape

    h_k = h // window_size
    w_k = w // window_size
    print('Window size:', window_size)

    for i in range(h_k):
        for j in range(w_k):
            window = img_points[i * window_size: (i + 1) * window_size, j * window_size: (j + 1) * window_size]
            if np.sum(window) > 1:
                # Keep only the first point in the window
                found_first = False
                for y in range(window.shape[0]):
                    for x in range(window.shape[1]):
                        if window[y, x] == 1 and not found_first:
                            found_first = True
                            print('Found first point:', (x + j * window_size, y + i * window_size))
                        else:
                            window[y, x] = 0

    return img_points


def ccw(A, B, C):
    ay, ax = A
    by, bx = B
    cy, cx = C
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def intersection_point(p0, p1, q0, q1):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = q0
    x3, y3 = q1

    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x3 - x2
    dy2 = y3 - y2

    denominator = dx1 * dy2 - dy1 * dx2

    # Se sono garantiti incidenti, non serve controllare il denominatore
    t = ((x2 - x0) * dy2 - (y2 - y0) * dx2) / denominator

    # Punto di intersezione sulla retta p0 + t * (p1 - p0)
    ix = x0 + t * dx1
    iy = y0 + t * dy1

    return int(ix), int(iy)


def extend_line(x1, y1, x2, y2, width, height):
    if x1 == x2:
        # Linea verticale: estendi dall'alto al basso
        return x1, 0, x2, height, 0, 255, 0
    elif y1 == y2:
        # Linea orizzontale: estendi da sinistra a destra
        return 0, y1, width, y2, 255, 0, 0
    else:
        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1

        if abs(m) > 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        points = []

        # Intersezione con bordo sinistro (x = 0)
        y_left = int(q)
        if 0 <= y_left <= height:
            points.append((0, y_left))

        # Intersezione con bordo destro (x = width)
        y_right = int(m * width + q)
        if 0 <= y_right <= height:
            points.append((width, y_right))

        # Intersezione con bordo superiore (y = 0)
        x_top = int(-q / m)
        if 0 <= x_top <= width:
            points.append((x_top, 0))

        # Intersezione con bordo inferiore (y = height)
        x_bottom = int((height - q) / m)
        if 0 <= x_bottom <= width:
            points.append((x_bottom, height))

        # Scegli i due punti più validi
        if len(points) >= 2:
            return *points[0], *points[1], *color
        else:
            return x1, y1, x2, y2, *color  # fallback



def extend_line_partial(x1, y1, x2, y2, width, height, scale=1.75):
    # The two line center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Direction along x and y
    dx = x2 - x1
    dy = y2 - y1

    #
    length = math.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2, 0, 0, 255  # punto singolo

    # Normalizza il vettore
    dx /= length
    dy /= length

    # I extend from the
    new_length = length * scale / 2
    x1_ext = cx - dx * new_length
    y1_ext = cy - dy * new_length
    x2_ext = cx + dx * new_length
    y2_ext = cy + dy * new_length

    # Clamping
    x1_ext = max(0, min(width, x1_ext))
    y1_ext = max(0, min(height, y1_ext))
    x2_ext = max(0, min(width, x2_ext))
    y2_ext = max(0, min(height, y2_ext))

    # Coloring for classification
    m = dy / dx if dx != 0 else float('inf')
    color = (0, 255, 0) if abs(m) > 1 else (255, 0, 0)

    return int(x1_ext), int(y1_ext), int(x2_ext), int(y2_ext), *color



def oblique(x1, y1, x2, y2, threshold):
    """
    Check if the line is oblique
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: True if the line is oblique, False otherwise
    """
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length == 0:
        return False

    if abs(((x2 - x1) / length) > threshold and ((y2 - y1) / length) > threshold):
        return True
    return False