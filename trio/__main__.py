import numpy as np
import cv2 as cv

from .image.tracking_buffer import TrackingBuffer


def uv_to_int(uv):
    return (int(round(uv[0])), int(round(uv[1])))


def run_app():
    cap = cv.VideoCapture(cv.samples.findFile("vtest.avi"))
    if not cap.isOpened():
        print("Failed to open video")
        exit()

    tb = TrackingBuffer()

    uv = (384.0, 288.0)
    uv2 = uv

    cv.namedWindow("Oldest")
    cv.namedWindow("Newest")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Bye")
            break

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
    run_app()
