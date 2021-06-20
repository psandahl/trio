import numpy as np
import numpy.linalg as linalg

import cv2 as cv
import scipy.optimize as opt

import functools
import json
import math

from .common.camera import Camera, Permutation
from .common.linear import solve_dlt
from .common.math import euclidean, homogeneous
from .common.matrix import matrix_intrinsic, matrix_permute_ned, \
    matrix_decompose_projection, matrix_ypr, matrix_decompose_ypr


def obj_from_file(path):
    return json.load(open(path))


def camera_from_parameters(param):
    position = np.array((param["x"], param["y"], param["z"]))
    orientation = np.radians((param["yaw"], param["pitch"], param["roll"]))
    fov = np.radians((param["horizontal-fov"], param["vertical-fov"]))
    return Camera(position, orientation, fov,
                  rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                  perm=Permutation.NED)


def intrinsic_and_permute(param):
    intrinsic = matrix_intrinsic(np.radians((param["horizontal-fov"],
                                             param["vertical-fov"])),
                                 rect=np.array([-0.5, -0.5, 1.0, 1.0]))
    return (intrinsic, matrix_permute_ned())


def reprojection_errors(points, camera):
    err = []

    for point in points:
        uv0 = np.array((point["u"], point["v"]))
        xyz = np.array((point["x"], point["y"], point["z"]))
        uv1 = camera.project(xyz)
        err.append(linalg.norm(uv0 - uv1))

    return err


def squared_sum(xs):
    return functools.reduce(lambda acc, x: acc + x**2, xs)


def obj_f(points, position=np.array((0, 0, 0)), orientation=np.array((0, 0, 0)),
          fov=np.array((0, 0))):
    camera = Camera(position, orientation, fov,
                    rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                    perm=Permutation.NED)
    return reprojection_errors(points, camera)


def minimize_orientation(points, position, orientation, fov):
    obj = functools.partial(obj_f, points, position=position, fov=fov)

    res = opt.least_squares(lambda ypr: obj(
        orientation=ypr), orientation, method='lm')

    return res.x


def minimize_position(points, position, orientation, fov):
    obj = functools.partial(obj_f, points, orientation=orientation, fov=fov)

    res = opt.least_squares(lambda xyz: obj(
        position=xyz), position, method='lm')

    return res.x


def minimize_fov(points, position, orientation, fov):
    obj = functools.partial(
        obj_f, points, position=position, orientation=orientation)

    res = opt.least_squares(lambda hv: obj(fov=hv), fov, method='lm')

    return res.x


def minimize_all(points, position, orientation, fov):
    obj = functools.partial(obj_f, points)

    x0 = np.zeros(8)
    np.put(x0, [0, 1, 2], position)
    np.put(x0, [3, 4, 5], orientation)
    np.put(x0, [6, 7], fov)

    res = opt.least_squares(lambda x: obj(
        position=x[0:3], orientation=x[3:6], fov=x[6:8]), x0, method='lm')

    return (res.x[0:3], res.x[3:6], res.x[6:8])


def run2(path):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] > 0.99:
            print("Image id: %d" % image["image-id"])
            print("confidence: %.5f" % image["confidence"])

            params = image["camera-parameters"]

            print("Camera params:\n%s" % params)

            position0 = np.array((params["x"], params["y"], params["z"]))
            orientation0 = np.array(
                (params["yaw"], params["pitch"], params["roll"]))
            fov0 = np.array((params["horizontal-fov"], params["vertical-fov"]))

            cam0 = Camera(position0, np.radians(orientation0), np.radians(fov0),
                          rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                          perm=Permutation.NED)
            intrinsic, permute = intrinsic_and_permute(params)

            cam0_ypr, cam0_t = matrix_decompose_projection(
                cam0.projection_matrix, intrinsic, permute)

            cam0_r = (matrix_ypr(np.array(cam0_ypr)) @ permute).T

            print("Cam0 y: %f, p: %f, r: %f" % (math.degrees(
                cam0_ypr[0]), math.degrees(cam0_ypr[1]), math.degrees(cam0_ypr[2])))
            print("Cam0 t: %s" % cam0_t)
            print("Cam0 r:\n%s" % cam0_r)

            print("Cam0 camera matrix:\n%s" % cam0.camera_matrix)

            print("===")

            points = image["point-correspondences"]
            obj_points = []
            img_points = []
            for point in points:
                obj_points.append((point["x"], point["y"], point["z"]))
                img_points.append((point["u"], point["v"]))

            # Solve EPNP
            ret, rvec, tvec = cv.solvePnP(np.array(obj_points), np.array(img_points),
                                          intrinsic, np.array([]),
                                          np.array([]), np.array([]),
                                          useExtrinsicGuess=False,
                                          flags=cv.SOLVEPNP_EPNP)

            print("Solve EPNP")
            print("tvec: %s" % tvec)
            rr, j = cv.Rodrigues(rvec)
            print("r:\n%s" % rr)

            # Solve SQPNP
            ret, rvec, tvec = cv.solvePnP(np.array(obj_points), np.array(img_points),
                                          intrinsic, np.array([]),
                                          np.array([]), np.array([]),
                                          useExtrinsicGuess=False,
                                          flags=cv.SOLVEPNP_SQPNP)

            print("Solve SQPNP")
            print("tvec: %s" % tvec)
            rr, j = cv.Rodrigues(rvec)
            print("r:\n%s" % rr)

            input("Press ENTER to continue")


def run(path):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] > 0.99:

            intrinsic, permute = intrinsic_and_permute(
                image["camera-parameters"])

            params = image["camera-parameters"]

            print("===")

            points = image["point-correspondences"]
            position0 = np.array((params["x"], params["y"], params["z"]))
            orientation0 = np.array(
                (params["yaw"], params["pitch"], params["roll"]))
            fov0 = np.array((params["horizontal-fov"], params["vertical-fov"]))

            print("Truth params.\n Position: %s\n Orientation: %s\n FOV: %s" %
                  (position0, orientation0, fov0))

            # Truth camera.
            cam0 = Camera(position0, np.radians(orientation0), np.radians(fov0),
                          rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                          perm=Permutation.NED)
            print("Truth err: %f" % squared_sum(
                reprojection_errors(points, cam0)))

            position1 = position0 + (32.0, 22.6, 4.5)
            print("Modified position: %s" % position1)

            orientation1 = orientation0 + (-3.31, 1.27, 1.0)
            print("Modified orientation: %s" % orientation1)

            fov1 = fov0 * 1.09
            print("Modified fov: %s" % fov1)

            print("===")

            """
            cam1 = Camera(position, np.radians(orientation1), np.radians(fov),
                          rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                          perm=Permutation.NED)
            print("Cam1 err: %f" % squared_sum(
                reprojection_errors(points, cam1)))
            """

            """
            cam11 = Camera(position, np.radians(orientation11), np.radians(fov),
                           rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                           perm=Permutation.NED)
            print("Cam11 err: %f" % squared_sum(
                reprojection_errors(points, cam11)))
            """

            position11 = minimize_position(
                points, position1, np.radians(orientation0), np.radians(fov0))

            print("Solved position11: %s" % position11)

            orientation11 = np.degrees(minimize_orientation(points, position0, np.radians(
                orientation1), np.radians(fov0)))

            print("Solved orientation11: %s" % orientation11)

            fov11 = np.degrees(minimize_fov(points,  position0, np.radians(
                orientation0), np.radians(fov1)))
            print("Solved fov11: %s" % fov11)

            (position12, orientation12, fov12) = minimize_all(
                points, position1, np.radians(orientation1), np.radians(fov1))

            print("Solved position12: %s" % position12)
            print("Solved orientation12: %s" % np.degrees(orientation12))
            print("Solved fov12: %s" % np.degrees(fov12))

            position13 = minimize_position(
                points, position12, orientation12, fov12)

            print("Solved position13: %s" % position13)

            orientation13 = np.degrees(minimize_orientation(
                points, position13, orientation12, fov12))

            print("Solved orientation13: %s" % orientation13)

            fov13 = np.degrees(minimize_fov(
                points, position13, np.radians(orientation13), fov12))
            print("Solved fov13: %s" % fov13)

            break
