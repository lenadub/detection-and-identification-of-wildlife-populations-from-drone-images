# src/data/preprocessing.py

import cv2
import numpy as np


def preprocess_image(image, size=(640, 640)):
    """
    Resize and normalize an image.

    Args:
        image (np.ndarray): RGB image
        size (tuple): target size (width, height)

    Returns:
        np.ndarray: preprocessed image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("preprocess_image expects a NumPy array")

    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image
