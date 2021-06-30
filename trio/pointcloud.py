import numpy as np
import cv2 as cv

import json

from .common.camera import Camera, Permutation, camera_from_param
from .image.matching_buffer import MatchingBuffer


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

    buffer = MatchingBuffer(15)

    index = 0
    while True:
        if index == len(frames):
            print("Reached end of frames array")
            break

        ret, image = eager_cap_read(cap)
        if not ret:
            print("Failed to receive video image")
            break

        frame = frames[index]
        index += 1

        height, width, channels = image.shape
        camera = camera_from_param(frame["camera-parameters"],
                                   rect=np.array(
                                       [0, 0, width - 1, height - 1]),
                                   perm=Permutation.NED)
        points = frame["point-correspondences"]
        confidence = frame["confidence"]
        buffer.push(image, camera, points, confidence)
        if not buffer.has_valid_pairing():
            print("Filling buffer")
            continue

        entry0, entry1 = buffer.valid_pairing()

        cv.imshow("Player", entry0["clean-image"])

        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
