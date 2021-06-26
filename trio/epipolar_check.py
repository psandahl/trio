import cv2 as cv
import numpy as np

import json

from .common.camera import Camera, Permutation, camera_from_param, \
    camera_fundamental_matrix
from .common.math import epipolar_line, plot_on_line

image_width = 1280
image_height = 720

selected_uvs = [(0., 0.), (-.25, -.25), (.25, -.25), (-.25, .25), (.25, .25)]

colors = [(255, 255, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 0, 0)]


def obj_from_file(path):
    return json.load(open(path))


def within_uv_selection(uv):
    for s in selected_uvs:
        if np.all(np.isclose(s, uv)):
            return True

    return False


def get_selected_points(points):
    selection = []
    for point in points:
        if within_uv_selection((point["u"], point["v"])):
            selection.append(np.array([point["x"], point["y"], point["z"]]))

    return selection


def uv_to_int(uv):
    u, v = uv
    return (int(round(u)), int(round(v)))


def process_frames(frame0, frame1):
    camera0 = camera_from_param(frame0["camera-parameters"],
                                rect=np.array(
        [0., 0., image_width - 1, image_height - 1]),
        perm=Permutation.NED)

    camera1 = camera_from_param(frame1["camera-parameters"],
                                rect=np.array(
        [0., 0., image_width - 1, image_height - 1]),
        perm=Permutation.NED)

    F = camera_fundamental_matrix(camera0, camera1)

    display = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    points = get_selected_points(frame0["point-correspondences"])
    for index in range(len(points)):
        point = points[index]
        color = colors[index]
        uv0 = camera0.project(point)

        l = epipolar_line(F, uv0)
        start_line = (0, int(round(plot_on_line(l, 0))))
        end_line = (image_width - 1,
                    int(round(plot_on_line(l, image_width - 1))))
        cv.line(display, start_line, end_line, color)

        cv.drawMarker(display, uv_to_int(uv0), color)

        uv1 = camera1.project(point)
        cv.circle(display, uv_to_int(uv1), 5, color, 1, cv.LINE_AA)

    return display


def run(path):
    frames = obj_from_file(path)["images"]

    cv.namedWindow("Epipolar Check")

    spacing = 1
    index = 0
    max_index = len(frames) - 1
    while True:
        frame0 = frames[index]
        frame1 = frames[index + spacing]

        print("Process frames '%d' and '%d'" %
              (frame0["image-id"], frame1["image-id"]))

        #print("Quit using ESC or 'q' - any other key step one frame")

        display = process_frames(frame0, frame1)
        cv.imshow("Epipolar Check", display)

        key = cv.waitKey(0)
        if key == 32 or key == ord('n'):
            index = min(max_index, index + 1)
        elif key == ord('p'):
            index = max(0, index - 1)
        elif key == ord('1'):
            spacing = 1
        elif key == ord('2'):
            spacing = 2
        elif key == ord('3'):
            spacing = 3
        elif key == ord('4'):
            spacing = 4
        elif key == ord('5'):
            spacing = 5
        elif key == ord('6'):
            spacing = 6
        elif key == 27 or key == ord('q'):
            break

    cv.destroyAllWindows()
