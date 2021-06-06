import numpy as np
import cv2 as cv

import json

from .common.camera import Camera, Permutation
from .common.linear import triangulate, solve_dlt
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


def projection_from_image(image, width, height):
    points = image["point-correspondences"]
    for point in points:
        point["u"] = (point["u"] + 0.5) * float(width - 1)
        point["v"] = (point["v"] + 0.5) * float(height - 1)

    res, p = solve_dlt(points)
    if not res:
        print("Failed to perform DLT")

    return p


def calc_depth_image(tb, aoi):
    xstart = aoi[0]
    ystart = aoi[1]
    width = aoi[2]
    height = aoi[3]

    d = np.zeros((height, width), dtype=np.float)

    p0 = tb.oldest_projection()
    p1 = tb.newest_projection()
    c = tb.oldest_camera()

    for row in range(height):
        for col in range(width):
            uv = (float(xstart + col), float(ystart + row))
            uv2 = tb.track(uv)

            xyz = triangulate(p0, np.array(uv),
                              p1, np.array(uv2))

            zC = c.camera_space(xyz)[2]
            d[row, col] = zC

    min_val = np.min(d)
    max_val = np.max(d)
    val_range = max_val - min_val

    for row in range(height):
        for col in range(width):
            zC = d[row, col]
            d[row, col] = (zC - min_val) / val_range
            # print("min: %.2f max: %.2f range: %.2f z: %.2f zn: %.2f" %
            #      (min_val, max_val, val_range, zC, d[row, col]))

    return d


def run_app2():
    f = open("/home/patrik/test-data/test-meta.json")
    meta = json.load(f)
    images = meta["images"]

    cap = cv.VideoCapture("/home/patrik/test-data/test-video.ts")
    if not cap.isOpened():
        print("Failed to open video")
        exit()

    cv.namedWindow("Player")
    cv.namedWindow("Depth AOI")

    # Create the tracking buffer.
    tb = TrackingBuffer()

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

        # Get projection matrix from image metadata.
        p = projection_from_image(images[frame_count],
                                  frame.shape[1] - 1, frame.shape[0] - 1)

        # print(cam.projection_matrix)
        # print("--")
        # print(p)

        # Add image + projection.
        tb.add_image(frame, cam, p)

        #aoi = (580, 250, 250, 180)
        aoi = (920, 370, 150, 150)

        #uv = ((frame.shape[1] - 1) / 2.0, (frame.shape[0] - 1) / 2.0)
        #uv2 = tb.track(uv)

        image = np.array(tb.oldest_image())

        cv.rectangle(image, aoi, (0, 0, 255))

        cv.imshow("Player", image)
        depth_image = calc_depth_image(tb, aoi)
        cv.imshow("Depth AOI", depth_image)
        # else:
        #    cv.imshow("Player", frame)

        # Some reprojection tests.
        # for corr in images[frame_count]["point-correspondences"]:
        #    px = uv_to_int(cam.project(np.array([corr["x"],
        #                                         corr["y"], corr["z"]])))
        #    cv.drawMarker(frame, px, (255, 0, 0))

        print("Frame=%d" % frame_count)

        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run_app2()
