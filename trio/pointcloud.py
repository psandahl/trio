import numpy as np
import cv2 as cv

import json

from .image.utils import remove_symbols


def eager_cap_read(cap):
    for n in range(50):
        ret, frame = cap.read()
        if ret:
            return (ret, frame)

    return (False, None)


def obj_from_file(path):
    return json.load(open(path))


def run(video_path, meta_path):
    frames = obj_from_file(meta_path)["images"]

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video '%s'" % video_path)
        return

    cv.namedWindow("Player")

    while True:
        ret, image = eager_cap_read(cap)
        if not ret:
            print("Failed to receive video image")
            break

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        clean = remove_symbols(gray)

        cv.imshow("Player", clean)

        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
