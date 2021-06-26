import cv2 as cv
import numpy as np

import json

from .common.camera import Camera, Permutation, camera_from_param, \
    camera_fundamental_matrix
from .common.math import epipolar_line, plot_on_line

from .common.matrix import matrix_rank

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
    # print("frame0: %.3f frame1: %.3f" %
    #      (frame0["confidence"], frame1["confidence"]))
    camera0 = camera_from_param(frame0["camera-parameters"],
                                rect=np.array(
        [0., 0., image_width - 1, image_height - 1]),
        perm=Permutation.NED)

    camera1 = camera_from_param(frame1["camera-parameters"],
                                rect=np.array(
        [0., 0., image_width - 1, image_height - 1]),
        perm=Permutation.NED)

    F = camera_fundamental_matrix(camera0, camera1)
    #print("F:\n%s" % F)
    print("rank(F): %d" % matrix_rank(F))

    display = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    points = get_selected_points(frame0["point-correspondences"])
    for index in range(len(points)):
        point = points[index]
        color = colors[index]
        uv0 = camera0.project(point)

        l = epipolar_line(F, uv0)
        start_line = (0, int(round(plot_on_line(l, 0))))
        end_line = (1279, int(round(plot_on_line(l, 1279))))
        cv.line(display, start_line, end_line, color)

        print("epipolar line: %s" % l)
        print("p(0): %f" % plot_on_line(l, 0))
        print("p(1279): %f" % plot_on_line(l, 1279))

        cv.drawMarker(display, uv_to_int(uv0), color)

        uv1 = camera1.project(point)
        cv.circle(display, uv_to_int(uv1), 5, color, 1, cv.LINE_AA)

    return display


def run(path):
    frames = obj_from_file(path)["images"]

    cv.namedWindow("Epipolar Check")

    index = 0
    max_index = len(frames) - 1
    while index < max_index:
        frame0 = frames[index]
        frame1 = frames[index + 1]
        index += 1

        print("Process frames '%d' and '%d'" %
              (frame0["image-id"], frame1["image-id"]))
        print("Quit using ESC or 'q' - any other key step one frame")

        display = process_frames(frame0, frame1)
        cv.imshow("Epipolar Check", display)

        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    cv.destroyAllWindows()
