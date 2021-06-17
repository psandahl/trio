import numpy as np
import numpy.linalg as linalg

import scipy.optimize as opt

from functools import partial
import json

from .common.camera import Camera, Permutation
from .common.linear import solve_dlt
from .common.math import euclidean, homogeneous
from .common.matrix import matrix_intrinsic, matrix_permute_ned, \
    matrix_decompose_projection


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


def project(projection, point):
    return euclidean(projection @ homogeneous(np.array(point)))


def reprojection_errors(points, xs):
    projection = xs.reshape(3, 4)
    errors = []
    for point in points:
        xyz = np.array((point["x"], point["y"], point["z"]))
        uv0 = np.array((point["u"], point["v"]))
        uv1 = project(projection, xyz)
        errors.append(linalg.norm(uv1 - uv0))

    return errors


def minimize(points, projection):
    obj_func = partial(reprojection_errors, points)
    res = opt.least_squares(
        obj_func, projection.flatten(), method='lm')
    if res.success:
        print(np.isclose(projection.flatten(), res.x))
        return res.x.reshape(3, 4)
    else:
        print("minimization failed")
        return projection


def run(path):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] > 0.99:
            print("Found image id: %d" % image["image-id"])
            print("Camera parameters: %s" % image["camera-parameters"])
            print("===")

            intrinsic, permute = intrinsic_and_permute(
                image["camera-parameters"])

            camera = camera_from_parameters(image["camera-parameters"])
            print("Camera projection matrix:\n%s" % camera.projection_matrix)

            cam_ypr, cam_t = matrix_decompose_projection(
                camera.projection_matrix, intrinsic, permute)

            print("Deconstructed camera ypr: %s\n camera translate: %s" %
                  (np.degrees(cam_ypr), cam_t))

            print("===")

            points = image["point-correspondences"]

            res, solve_projection = solve_dlt(points)

            print("Solved projection matrix:\n%s" % solve_projection)

            solve_ypr, solve_t = matrix_decompose_projection(
                solve_projection, intrinsic, permute)

            print("Deconstructed solve ypr: %s\n solve translate: %s" %
                  (np.degrees(solve_ypr), solve_t))

            print("===")

            err0 = reprojection_errors(
                points, camera.projection_matrix.flatten())
            err1 = reprojection_errors(points, solve_projection.flatten())

            print("Camera reprojection error sum: %f" % sum(err0))
            print("Min: %.12f, Max: %.12f" % (np.min(err0), np.max(err0)))
            print("Solved reprojection error sum: %f" % sum(err1))
            print("Min: %.12f, Max: %.12f" % (np.min(err1), np.max(err1)))

            print("===")

            cam_minimized = minimize(points, camera.projection_matrix)
            solve_minimized = minimize(points, solve_projection)

            err00 = reprojection_errors(points, cam_minimized.flatten())
            err11 = reprojection_errors(points, solve_minimized.flatten())

            print("Camera minimized reprojection error sum: %f" % sum(err00))
            print("Min: %.12f, Max: %.12f" % (np.min(err00), np.max(err00)))
            print("Solved minimized reprojection error sum: %f" % sum(err11))
            print("Min: %.12f, Max: %.12f" % (np.min(err11), np.max(err11)))

            break
