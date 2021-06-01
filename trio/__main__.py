import numpy as np
import cv2 as cv


def run_app():
    cap = cv.VideoCapture(cv.samples.findFile("vtest.avi"))
    if not cap.isOpened():
        print("Failed to open video")
        exit()

    cv.namedWindow("Player")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Bye")
            break

        cv.imshow("Player", frame)
        key = cv.waitKey(30)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_app()
