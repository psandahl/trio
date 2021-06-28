import cv2 as cv
import numpy as np

import json

from .common.camera import Camera, Permutation, camera_from_param, \
    camera_fundamental_matrix
from .common.linear import closest_point_on_line, triangulate
from .common.math import epipolar_line, plot_on_line

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

    max_line_err, max_tri_err = .0, .0

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

        # Calulate the line error.
        line_err = np.linalg.norm(pt - uv1)
        max_line_err = max(line_err, max_line_err)

        if line_err > 1.0:
            # Draw a line between the point and the uv coordinate.
            cv.line(display, uv_to_int(uv1),
                    uv_to_int(pt), color, 1, cv.LINE_AA)

        # Triangulate the two uv coordinates and measure their distance.
        tri_point = triangulate(camera0.projection_matrix, uv0,
                                camera1.projection_matrix, uv1).flatten()
        tri_err = np.linalg.norm(tri_point - point)
        max_tri_err = max(tri_err, max_tri_err)

    text = "Max epiline err: {:.5f}, max triangulation err: {:.3f}m".format(
        max_line_err, max_tri_err)
    text_size, baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    text_x_start = int(image_width / 2 - text_size[0] / 2)
    cv.putText(display, text, (text_x_start, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

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

        print("Quit using ESC or 'q' - SPACE or 'n' forward - 'p' backward")
        print("'1' to '9' to set spacing. Current spacing: %d" % spacing)

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
        elif key == ord('7'):
            spacing = 7
        elif key == ord('8'):
            spacing = 8
        elif key == ord('9'):
            spacing = 9
        elif key == 27 or key == ord('q'):
            break

    cv.destroyAllWindows()


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
