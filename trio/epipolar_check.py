import cv2 as cv
import numpy as np

import json

from .common.camera import Camera, Permutation, camera_from_param, \
    camera_fundamental_matrix
from .common.linear import closest_point_on_line
from .common.math import epipolar_line, plot_on_line
from .common.matrix import matrix_look_at, matrix_decompose_ypr, matrix_ypr

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


def process_frames(camera0, camera1, points, image_width, image_height):

    F = camera_fundamental_matrix(camera0, camera1)

    display = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for index in range(len(points)):
        point = points[index]
        color = colors[index]

        # This is the 3d point projected in camera 0.
        uv0 = camera0.project(point)
        cv.drawMarker(display, uv_to_int(uv0), color)

        # This is the 3d point projected in camera 1.
        uv1 = camera1.project(point)
        cv.circle(display, uv_to_int(uv1), 5, color, 1, cv.LINE_AA)

        # Calculate the epipolar line for camera 1 from the uv coordinate for
        # camera 0.
        line = epipolar_line(F, uv0)

        # Plot the epipolar line.
        start_line = uv_to_int((0, plot_on_line(line, 0)))
        end_line = uv_to_int((image_width - 1,
                              plot_on_line(line, image_width - 1)))

        cv.line(display, start_line, end_line, color, 1, cv.LINE_AA)

        # Find the closest point on the epipolar line for the uv from camera 1.
        pt = closest_point_on_line(line, uv1)
        disp = np.linalg.norm(pt - uv1)

        if disp > 1.0:
            # Draw a line between the point and the uv coordinate.
            cv.line(display, uv_to_int(uv1),
                    uv_to_int(pt), color, 1, cv.LINE_AA)

    return display


def run(path, image_width=1280, image_height=720):
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

        # print("Quit using ESC or 'q' - any other key step one frame")

        param0 = frame0["camera-parameters"]
        param1 = frame1["camera-parameters"]
        camera0 = camera_from_param(param0,
                                    rect=np.array(
                                        [0., 0., image_width - 1, image_height - 1]),
                                    perm=Permutation.NED)

        camera1 = camera_from_param(param1,
                                    rect=np.array(
                                        [0., 0., image_width - 1, image_height - 1]),
                                    perm=Permutation.NED)
        points = get_selected_points(frame0["point-correspondences"])

        display = process_frames(
            camera0, camera1, points, image_width, image_height)
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


def param_from_data(pos, ypr, fov):
    param = dict()

    x, y, z = pos
    yaw, pitch, roll = ypr
    h, v = fov

    param["x"] = x
    param["y"] = y
    param["z"] = z
    param["yaw"] = yaw
    param["pitch"] = pitch
    param["roll"] = roll
    param["horizontal-fov"] = h
    param["vertical-fov"] = v

    return param


def run_stereo_normal():
    image_width = 1280
    image_height = 720
    fov = (30, 20)

    points = [
        np.array([11.0, 0, 2.0]),
        np.array([8.0, 0, 1.5]),
        np.array([9.5, 0, 1.0]),
        np.array([11.0, 0, 0.5]),
        np.array([8.0, 0, 0.0])
    ]

    camera0 = Camera(np.array((10, 10, 1)), np.radians((-90, 0, 0)),
                     np.radians(fov), rect=np.array(
                         [0., 0., image_width - 1, image_height - 1]),
                     perm=Permutation.ECEF)

    camera1 = Camera(np.array((9, 9, 1)), np.radians((-88, 0, 0)),
                     np.radians(fov), rect=np.array(
                         [0., 0., image_width - 1, image_height - 1]),
                     perm=Permutation.ECEF)

    F = camera_fundamental_matrix(camera0, camera1)

    print("F:\n%s" % F)

    display = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv.namedWindow("Stereo Normal")

    for i in range(len(points)):
        uv0 = camera0.project(points[i])
        cv.drawMarker(display, uv_to_int(uv0), colors[i])

        uv1 = camera1.project(points[i])
        cv.circle(display, uv_to_int(uv1), 5, colors[i], 1, cv.LINE_AA)

        line = epipolar_line(F, uv0)
        print("line: %s" % line)

        # Plot the epipolar line.
        start_line = (0, int(round(plot_on_line(line, 0))))
        end_line = (image_width - 1,
                    int(round(plot_on_line(line, image_width - 1))))
        cv.line(display, start_line, end_line, colors[i], 1, cv.LINE_AA)

    cv.imshow("Stereo Normal", display)
    cv.waitKey(0)

    cv.destroyAllWindows()


def run_synthetic():
    image_width = 1280
    image_height = 720
    height = 50.0
    at_points = [
        np.array([6.0, 6.0, 0.0]),
        np.array([5.0, 5.0, 0.0]),
        np.array([4.0, 4.0, 0.0]),
        np.array([3.0, 3.0, 0.0]),
        np.array([2.0, 2.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0])
    ]
    eye_points = [
        np.array([84.0, 84.0, height]),
        np.array([82.0, 82.0, height]),
        np.array([80.0, 80.0, height]),
        np.array([78.0, 78.0, height]),
        np.array([76.0, 76.0, height]),
        np.array([74.0, 74.0, height]),
        np.array([72.0, 72.0, height])
    ]
    eye_points2 = [
        np.array([84.0, 70.0, height]),
        np.array([82.0, 70.0, height]),
        np.array([80.0, 70.0, height]),
        np.array([78.0, 70.0, height]),
        np.array([76.0, 70.0, height]),
        np.array([74.0, 70.0, height]),
        np.array([72.0, 70.0, height])
    ]
    ground_points = [
        np.array([-3.5, -7.0, 10.0]),
        np.array([10.0, -10.0, 6.0]),
        np.array([0.0, 0.0, 7.0]),
        np.array([-10.0, 10.0, 8.0]),
        np.array([6.0, 5.0, 8.0]),
    ]

    up = np.array((0.0, 0.0, 1.0))
    fov = (30, 20)

    cv.namedWindow("Epipolar Check")

    for i in range(len(at_points) - 1):
        ypr0 = np.degrees(matrix_decompose_ypr(
            matrix_look_at(eye_points[i], at_points[i], up)))
        param0 = param_from_data(eye_points[i], ypr0, fov)
        camera0 = camera_from_param(param0,
                                    rect=np.array(
                                        [0., 0., image_width - 1, image_height - 1]),
                                    perm=Permutation.ECEF)

        ypr1 = np.degrees(matrix_decompose_ypr(
            matrix_look_at(eye_points[i + 1], at_points[i + 1], up)))
        param1 = param_from_data(eye_points[i + 1], ypr1, fov)
        camera1 = camera_from_param(param1,
                                    rect=np.array(
                                        [0., 0., image_width - 1, image_height - 1]),
                                    perm=Permutation.ECEF)

        display = process_frames(
            camera0, camera1, ground_points, image_width, image_height)

        cv.imshow("Epipolar Check", display)
        cv.waitKey(0)

    cv.destroyAllWindows()
