import cv2 as cv
import numpy as np
import os
import argparse
from utils import timer

def resize_img(img, TARGET=512):
    height, width = img.shape[:2]

    TARGET = 512
    lmax = max(height, width)
    lmin = min(height, width)

    ratio = float(lmin) / float(lmax)

    l_max = TARGET
    l_min = int(TARGET * ratio)

    if max == height:
        return cv.resize(img, (l_max, l_min), interpolation=cv.INTER_LINEAR)
    else:
        return cv.resize(img, (l_min, l_max), interpolation=cv.INTER_LINEAR)

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

def callback():
    pass


def houghTune():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'img', 'cropped-100-163-303-454.jpg')
    img = cv.imread(imgPath)

    height, width = img.shape[:2]
    heightScale = 768 / height
    widthScale = 1024 / width

    img = cv.resize(img, (int(widthScale * width), int(heightScale * height)), interpolation=cv.INTER_LINEAR)

    winnowName = 'Hough Transform'
    cv.namedWindow(winnowName)
    cv.createTrackbar('thresh', winnowName, 60, 255, callback)
    cv.createTrackbar('min-len', winnowName, 45, 255, callback)
    cv.createTrackbar('max-gap', winnowName, 5, 255, callback)
    cv.createTrackbar('oblique-thresh', winnowName, 100, 255, callback)

    paramWindow = 'Smoothing and Canny'
    cv.namedWindow(paramWindow)
    cv.createTrackbar('Sigma', paramWindow, 2, 10, callback)
    cv.createTrackbar('Kernel Size', paramWindow, 5, 10, callback)
    cv.createTrackbar('Low Threshold', paramWindow, 85, 255, callback)
    cv.createTrackbar('High Threshold', paramWindow, 239, 255, callback)
    cv.createTrackbar('L2 Gradient', paramWindow, 1, 1, callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        threshold = cv.getTrackbarPos('thresh', winnowName)
        minLineLength = cv.getTrackbarPos('min-len', winnowName)
        maxLineGap = cv.getTrackbarPos('max-gap', winnowName)
        olbiqueThreshold = cv.getTrackbarPos('oblique-thresh', winnowName) * 0.01

        sigma = cv.getTrackbarPos('Sigma', paramWindow)
        kernelSize = cv.getTrackbarPos('Kernel Size', paramWindow)
        if kernelSize % 2 == 0:
            kernelSize += 1
            cv.setTrackbarPos('Kernel Size', paramWindow, kernelSize)
        lowThreshold = cv.getTrackbarPos('Low Threshold', paramWindow)
        highThreshold = cv.getTrackbarPos('High Threshold', paramWindow)
        L2gradient = cv.getTrackbarPos('L2 Gradient', paramWindow)
        if L2gradient == 0:
            L2gradient = False
        else:
            L2gradient = True


        # Gaussian blur
        img_mod = cv.GaussianBlur(img, (kernelSize, kernelSize), sigma)
        # Canny edge detection
        edges = cv.Canny(img_mod, lowThreshold, highThreshold, L2gradient=L2gradient)

        edges = (edges > 0).astype(np.uint8) * 255

        cv.namedWindow('Canny')
        cv.imshow('Canny', edges)

        # Hough Transform (probabilistic)
        lines = cv.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi / 180,
                               threshold=threshold,
                               minLineLength=minLineLength,
                               maxLineGap=maxLineGap)

        # Draw lines
        if lines is not None:
            edges_copy = edges.copy()
            edges_copy[edges_copy > 0] = 0
            for i in range(0, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                if oblique(x1, y1, x2, y2, threshold=olbiqueThreshold) or (2 == 1):
                    pass
                else:
                    cv.line(edges_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv.imshow(winnowName, edges_copy)

def cannyTune():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'img', 'cropped-100-163-303-454.jpg')
    img = cv.imread(imgPath)

    img = resize_img(img)

    winnowName = 'Canny Edge Detection'
    cv.namedWindow(winnowName)
    cv.createTrackbar('Low Threshold', winnowName, 100, 255, callback)
    cv.createTrackbar('High Threshold', winnowName, 200, 255, callback)
    cv.createTrackbar('Gaussian Blur', winnowName, 2, 10, callback)
    cv.createTrackbar('Kernel Size', winnowName, 3, 10, callback)


    while True:
        if cv.waitKey(1) == ord('q'):
            break

        sigma = cv.getTrackbarPos('Gaussian Blur', winnowName)
        lowThreshold = cv.getTrackbarPos('Low Threshold', winnowName)
        highThreshold = cv.getTrackbarPos('High Threshold', winnowName)

        if sigma > 0:
            kernelSize = cv.getTrackbarPos('Kernel Size', winnowName)
            if kernelSize % 2 == 0:
                kernelSize += 1
            imgBlur = cv.GaussianBlur(img, (kernelSize, kernelSize), sigma)
        else:
            imgBlur = img

        edges = cv.Canny(imgBlur, lowThreshold, highThreshold)
        cv.imshow(winnowName, edges)

    cv.destroyAllWindows()




def contourTune():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'img', 'cropped-100-163-303-454.jpg')
    img = cv.imread(imgPath)

    img = resize_img(img)

    winnowName = 'Canny Edge Detection'
    cv.namedWindow(winnowName)
    cv.createTrackbar('Low Threshold', winnowName, 100, 255, callback)
    cv.createTrackbar('High Threshold', winnowName, 200, 255, callback)
    cv.createTrackbar('Gaussian Blur', winnowName, 2, 10, callback)
    cv.createTrackbar('Kernel Size', winnowName, 3, 10, callback)


    while True:
        if cv.waitKey(1) == ord('q'):
            break

        sigma = cv.getTrackbarPos('Gaussian Blur', winnowName)
        lowThreshold = cv.getTrackbarPos('Low Threshold', winnowName)
        highThreshold = cv.getTrackbarPos('High Threshold', winnowName)

        if sigma > 0:
            kernelSize = cv.getTrackbarPos('Kernel Size', winnowName)
            if kernelSize % 2 == 0:
                kernelSize += 1
            imgBlur = cv.GaussianBlur(img, (kernelSize, kernelSize), sigma)
        else:
            imgBlur = img

        edges = cv.Canny(imgBlur, lowThreshold, highThreshold)

        edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        img_copy = img.copy()
        for contour in contours:
            cv.drawContours(img_copy, [contour], -1, (0, 255, 0), 2)

        cv.imshow(winnowName, img_copy)

    cv.destroyAllWindows()



def harrisTune():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'img', 'cropped-100-163-303-454.jpg')
    img = cv.imread(imgPath)

    img = resize_img(img)

    img_corner = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    paramWin = 'Smoothing parameters'
    cv.namedWindow(paramWin)
    cv.createTrackbar('Kernel-size', paramWin, 3, 11, callback)
    cv.createTrackbar('Sigma', paramWin, 2, 10, callback)

    winnowName = 'Harris Corner Detection'
    cv.namedWindow(winnowName)
    cv.createTrackbar('Block Size', winnowName, 2, 16, callback)
    cv.createTrackbar('K (sigma)', winnowName, 1, 3, callback)
    cv.createTrackbar('K', winnowName, 4, 6, callback)
    cv.createTrackbar('Suppression', winnowName, 1, 100, callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            blockSize = cv.getTrackbarPos('Block Size', winnowName) + 1
            ksize = cv.getTrackbarPos('K', winnowName) * 0.01
            k_sigma = cv.getTrackbarPos('K (sigma)', winnowName)
            suppression = cv.getTrackbarPos('Suppression', winnowName) * 0.01 + 0.1
            print(f'Best parameters: blockSize={blockSize}, ksize={ksize}, k_sigma={k_sigma}, suppression={suppression}')
            break

        blockSize = cv.getTrackbarPos('Block Size', winnowName) + 1
        ksize = cv.getTrackbarPos('K', winnowName) * 0.01
        k_sigma = cv.getTrackbarPos('K (sigma)', winnowName)
        suppression = cv.getTrackbarPos('Suppression', winnowName) * 0.01 + 0.1

        if k_sigma % 2 == 0:
            k_sigma += 1
            cv.setTrackbarPos('K (sigma)', winnowName, k_sigma)

        kernelSize = cv.getTrackbarPos('Kernel-size', paramWin)
        if kernelSize % 2 == 0:
            kernelSize += 1
            cv.setTrackbarPos('Kernel-size', paramWin, kernelSize)
        sigma = cv.getTrackbarPos('Sigma', paramWin)

        img_corner_mod = cv.GaussianBlur(img_corner, (kernelSize, kernelSize), sigma)
        cv.namedWindow('Gaussian Blur')
        cv.imshow('Gaussian Blur', img_corner_mod)

        corner = cv.cornerHarris(img_corner_mod,
                                 blockSize=blockSize,
                                 ksize=k_sigma,
                                 k=ksize)
        corner = cv.dilate(corner, None, iterations=1)

        img_copy = img.copy()
        img_copy[corner > suppression * corner.max()] = [0, 0, 255]
        cv.imshow(winnowName, img_copy)


def bilateralTune():
    root = os.getcwd()
    imgPath = os.path.join(root, 'data', 'img', 'art-gallery1.jpg')
    img = cv.imread(imgPath)

    height, width = img.shape[:2]
    height, width = img.shape[:2]
    heightScale = 768 / height
    widthScale = 1024 / width

    img = cv.resize(img, (int(widthScale * width), int(heightScale * height)), interpolation=cv.INTER_LINEAR)

    winnowName = 'Bilateral filter'
    cv.namedWindow(winnowName)
    cv.createTrackbar('d', winnowName, 5, 15, callback)
    cv.createTrackbar('sigmaColor', winnowName, 75, 255, callback)
    cv.createTrackbar('sigmaSpace', winnowName, 150, 255, callback)

    i = 0
    while True:
        if cv.waitKey(1) == ord('q'):
            break

        d = cv.getTrackbarPos('d', winnowName)
        sigmaColor = cv.getTrackbarPos('sigmaColor', winnowName)
        sigmaSpace = cv.getTrackbarPos('sigmaSpace', winnowName)

        if i % 100 == 0:
            with timer.Timer('didjidj'):
                bilateral = cv.bilateralFilter(img, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT)
        else:
            bilateral = cv.bilateralFilter(img, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT)
        i += 1
        edges = cv.Canny(bilateral, 85, 239)
        edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]

        new_img = img.copy()
        new_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:
                cv.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv.imshow(winnowName, new_img)

    cv.destroyAllWindows()







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Tuner for object detection")
    parser.add_argument("operation", help="Operation to tune", choices=["canny", "hough", "harris", "contour", "bilateral"])

    args = parser.parse_args()

    if args.operation == "canny":
        cannyTune()
    elif args.operation == "hough":
        houghTune()
    elif args.operation == "harris":
        harrisTune()
    elif args.operation == 'contour':
        contourTune()
    elif args.operation == 'bilateral':
        bilateralTune()


    cv.destroyAllWindows()
