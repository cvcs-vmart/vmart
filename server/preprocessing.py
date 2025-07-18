import cv2
import numpy as np


def calculate_sharpness(image):
    """Here we compute the sharpness of the image using Laplacian operator"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()  # Varianza come misura di nitidezza
    return sharpness


def calculate_blur(image):
    """Here we compute the grade of blur using Sobel operator"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    blur = np.sqrt(sobelx ** 2 + sobely ** 2).mean()
    return blur


def select_best_frame(frames):
    """Here we select the best frame to do object detection"""
    best_score = 0
    best_frame = None
    best_pose = None
    for frame, pose in frames:
        sharpness = calculate_sharpness(frame)
        blur = calculate_blur(frame)
        score = sharpness / (blur + 1e-6)  # Rapporto nitidezza/sfocatura

        if score > best_score:
            best_score = score
            best_frame = frame
            best_pose = pose

    return best_frame, best_pose, best_score

# best_frame, best_score = select_best_frame(frames)
