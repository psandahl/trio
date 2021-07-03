import numpy as np
import cv2 as cv

import json

from .common.camera import Camera, Permutation, camera_from_param, \
    camera_fundamental_matrix
from .common.linear import closest_point_on_line, triangulate
from .common.math import epipolar_line, plot_on_line
from .common.point_set import PointSet
from .image.matching_buffer import MatchingBuffer

selected_uvs = [(0., 0.), (-.25, -.25), (.25, -.25), (-.25, .25), (.25, .25)]

colors = [(255, 255, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 0, 0)]


def eager_cap_read(cap):
    for n in range(50):
        ret, frame = cap.read()
        if ret:
            return (ret, frame)

    return (False, None)


def obj_from_file(path):
    return json.load(open(path))


def uv_to_int(uv):
    u, v = uv
    return (int(round(u)), int(round(v)))


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


def image_width_and_height(a):
    s = a.shape
    if len(s) == 3:
        h, w, c = s
        return (w, h)
    elif len(s) == 2:
        h, w = s
        return (w, h)
    else:
        return (0, 0)


def display_epipolar_and_reprojection(F, entry0, entry1):
    camera0 = entry0["camera"]
    camera1 = entry1["camera"]

    display = np.array(entry0["orig-image"])
    image_width, image_height = image_width_and_height(display)

    points = entry0["points"]

    for i in range(len(points)):
        color = colors[i]
        point = points[i]

        # Draw camera 0 as circle.
        uv0 = camera0.project(point)
        cv.circle(display, uv_to_int(uv0), 5, color, 1, cv.LINE_AA)

        # Draw camera 1 as cross.
        uv1 = camera1.project(point)
        cv.drawMarker(display, uv_to_int(uv1), color)

        # Calculate the epipolar line for camera 1 from the uv coordinate for
        # camera 0.
        line = epipolar_line(F, uv0)

        # Plot the epipolar line.
        start_line = uv_to_int((0, plot_on_line(line, 0)))
        end_line = uv_to_int((image_width - 1,
                              plot_on_line(line, image_width - 1)))

        cv.line(display, start_line, end_line, color, 1, cv.LINE_AA)

    cv.imshow("Epipolar and reprojection", display)


def display_best_matches(entry0, entry1, matches, window):
    display = cv.drawMatches(entry0["orig-image"],
                             entry0["keypoints"],
                             entry1["orig-image"],
                             entry1["keypoints"],
                             matches, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow(window, display)


def display_some_matches_with_epipolar(F, entry0, entry1, matches, window):
    display = np.array(entry1["orig-image"])
    image_width, image_height = image_width_and_height(display)
    kpt1 = entry0["keypoints"]
    kpt2 = entry1["keypoints"]
    index = 0
    for match in matches[:5]:
        color = colors[index]
        index += 1

        uv0 = kpt1[match.queryIdx].pt
        uv1 = kpt2[match.trainIdx].pt

        # Calculate the epipolar line for uv0.
        line = epipolar_line(F, np.array(uv0))

        # Plot the epipolar line.
        start_line = uv_to_int((0, plot_on_line(line, 0)))
        end_line = uv_to_int((image_width - 1,
                              plot_on_line(line, image_width - 1)))

        cv.line(display, start_line, end_line, color, 1, cv.LINE_AA)

        # Plot uv1 as cross.
        cv.drawMarker(display, uv_to_int(uv1), color)

    cv.imshow(window, display)


def optimize_matches(F, entry0, entry1, matches, epi_thres):
    optimized = []

    kpt0 = entry0["keypoints"]
    kpt1 = entry1["keypoints"]
    for match in matches:
        uv0 = kpt0[match.queryIdx].pt
        uv1 = kpt1[match.trainIdx].pt

        # Calculate the epipolar line for uv0.
        line = epipolar_line(F, np.array(uv0))

        # Get the closest point.
        pt = closest_point_on_line(line, uv1)

        err = np.linalg.norm(np.array(uv1) - pt)
        if err < epi_thres:
            optimized.append(match)

    return optimized


def triangulate_matches(entry0, entry1, matches, point_set):
    camera0 = entry0["camera"]
    camera1 = entry1["camera"]
    kpt0 = entry0["keypoints"]
    kpt1 = entry1["keypoints"]

    for match in matches:
        uv0 = np.array(kpt0[match.queryIdx].pt)
        uv1 = np.array(kpt1[match.trainIdx].pt)

        point = triangulate(camera0.projection_matrix, uv0,
                            camera1.projection_matrix, uv1)
        point_set.add(point)


def reproject_points(entry, points, window):
    display = np.array(entry["orig-image"])
    camera = entry["camera"]

    for point in points:
        uv = camera.project(point).flatten()
        cv.drawMarker(display, uv_to_int(uv), (0, 255, 0))

    cv.imshow(window, display)


def process_pair(entry0, entry1, matches, epi_thres, point_set):
    F = camera_fundamental_matrix(entry0["camera"], entry1["camera"])
    display_epipolar_and_reprojection(F, entry0, entry1)

    optimized = optimize_matches(F, entry0, entry1, matches, epi_thres)

    display_best_matches(entry0, entry1, optimized, "Sorted matches")
    display_some_matches_with_epipolar(
        F, entry0, entry1, optimized, "Matched epipolar")

    triangulate_matches(entry0, entry1, optimized, point_set)
    #reproject_points(entry0, point_set.points, "All points")


def run(video_path, meta_path, point_dist=0.5, buffer_width=30, epi_thres=0.5):
    frames = obj_from_file(meta_path)["images"]

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video '%s'" % video_path)
        return

    cv.namedWindow("Epipolar and reprojection")
    cv.namedWindow("Sorted matches")
    cv.namedWindow("Matched epipolar")
    #cv.namedWindow("All points")

    matching_buffer = MatchingBuffer(buffer_width)
    point_set = PointSet(point_dist)

    print(len(point_set.points))
    print("???")

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
        points = get_selected_points(frame["point-correspondences"])
        confidence = frame["confidence"]
        matching_buffer.push(image, camera, points, confidence)
        if not matching_buffer.has_valid_pairing():
            print("Filling buffer")
            continue

        entry0, entry1, matches = matching_buffer.valid_pairing()
        process_pair(entry0, entry1, matches, epi_thres, point_set)
        print(len(point_set.points))

        key = cv.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
