import numpy as np
import cv2 as cv

import json

from .common.camera import Camera, Permutation
from .image.tracking_buffer import TrackingBuffer


def uv_to_int(uv):
    return (int(round(uv[0])), int(round(uv[1])))


def eager_read(cap):
    n = 0
    for n in range(5):
        ret, frame = cap.read()
        if ret:
            return (ret, frame)

    return (False, None)


def camera_from_image(image, width, height):
    params = image["camera-parameters"]

    c = Camera(np.array([params["x"], params["y"], params["z"]]),
               np.radians((params["yaw"], params["pitch"], params["roll"])),
               np.radians((params["horizontal-fov"], params["vertical-fov"])),
               np.array([0.0, 0.0, width, height]),
               Permutation.NED
               )

    return c


def run_app2():
    f = open("/home/patrik/test-data/test-meta.json")
    meta = json.load(f)
    images = meta["images"]

    cap = cv.VideoCapture("/home/patrik/test-data/test-video.ts")
    if not cap.isOpened():
        print("Failed to open video")
        exit()

    cv.namedWindow("Player")

    frame_count = 0
    while True:
        ret, frame = eager_read(cap)

        if not ret:
            print("Can't receive frame. Bye")
            continue

        if frame_count == len(images):
            print("No more meta frames. Bye")

        # Get camera from image metadata.
        cam = camera_from_image(images[frame_count],
                                frame.shape[1] - 1, frame.shape[0] - 1)

        # Some reprojection tests.
        for corr in images[frame_count]["point-correspondences"]:
            px = uv_to_int(cam.project(np.array([corr["x"],
                                                 corr["y"], corr["z"]])))
            cv.drawMarker(frame, px, (255, 0, 0))

        cv.imshow("Player", frame)

        key = cv.waitKey(10)
        if key == 27 or key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()


def run_app():
    cap = cv.VideoCapture("/home/patrik/test-data/test-video.ts")
    if not cap.isOpened():
        print("Failed to open video")
        exit()

    tb = TrackingBuffer()

    uv = (384.0, 288.0)
    uv2 = uv

    cv.namedWindow("Oldest")
    cv.namedWindow("Newest")
    while True:
        ret, frame = eager_read(cap)

        if not ret:
            print("Can't receive frame. Bye")
            continue

        tb.add_image(frame)

        uv2 = tb.track(uv2)

        oldest_image = np.array(tb.oldest_image())
        newest_image = np.array(tb.newest_image())

        cv.drawMarker(oldest_image, uv_to_int(uv), (255, 0, 0))
        cv.drawMarker(newest_image, uv_to_int(uv2), (0, 255, 0))

        cv.imshow("Oldest", oldest_image)
        cv.imshow("Newest", newest_image)
        key = cv.waitKey(10)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # run_app()
    run_app2()
