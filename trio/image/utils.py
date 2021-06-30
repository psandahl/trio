import numpy as np
import cv2 as cv


def remove_symbols(image):
    mask = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 9, 35)
    mask = cv.erode(mask, np.array([]))
    smooth = cv.dilate(image, np.array([]))
    smooth = cv.medianBlur(smooth, 3)

    return (image & mask) + (smooth & ~mask)
