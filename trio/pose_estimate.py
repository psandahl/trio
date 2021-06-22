from scipy import linalg, optimize
import numpy as np

import functools
import json
import math

from .common.camera import Camera, Permutation, \
    camera_from_param, camera_reprojection_errors
from .common.math import sad
from .common.matrix import matrix_permute_ned, matrix_intrinsic
from .common.linear import solve_pose_epnp


def obj_from_file(path):
    return json.load(open(path))


def intrinsic_and_permute(fov):
    intrinsic = matrix_intrinsic(fov, rect=np.array([-0.5, -0.5, 1.0, 1.0]))
    return (intrinsic, matrix_permute_ned())


def close_angles(theta0, theta1):
    return abs(theta0 - theta1) < 0.00001  # Close enough?


def equal_angles(yaw0, yaw1, pitch0, pitch1, roll0, roll1):
    return close_angles(yaw0, yaw1) and close_angles(pitch0, pitch1) and \
        close_angles(roll0, roll1)


def position_distance(pos0, param):
    return linalg.norm(pos0 - np.array((param["x"], param["y"], param["z"])))


def camera_from_parts(position, orientation, fov):
    return Camera(position, orientation, fov,
                  rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                  perm=Permutation.NED)


def obj_f(points, position=np.array([]), orientation=np.array([]), fov=np.array([])):
    camera = camera_from_parts(position, orientation, fov)
    return camera_reprojection_errors(points, camera)


def optimize_fov(points, position, orientation, fov):
    obj = functools.partial(obj_f, points, position=position,
                            orientation=orientation)
    res = optimize.least_squares(lambda x: obj(fov=x), fov, method='lm')
    return res.x


def compare_image(image, fov_error):
    image_id = image["image-id"]
    param = image["camera-parameters"]
    points = image["point-correspondences"]

    fov = np.radians((param["horizontal-fov"], param["vertical-fov"]))

    # Start by solving for pose.
    intrinsic, permute = intrinsic_and_permute(fov)
    ret, ypr, t = solve_pose_epnp(points, intrinsic, permute)
    if ret:
        # Convert solved rotation to degrees.
        yaw, pitch, roll = (math.degrees(ypr[0]),
                            math.degrees(ypr[1]),
                            math.degrees(ypr[2]))

        # Compare solved angles with metadata.
        if not equal_angles(yaw, param["yaw"], pitch, param["pitch"],
                            roll, param["roll"]):
            print("Solved angles are not equal enough for frame id: %d" % image_id)

        # Compare distance between solved position and metadata.
        if position_distance(t, param) > 1.0:
            print("Position distance exceeds limit for frame id: %d" % image_id)
            print("Position distance: %f" % position_distance(t, param))

        # Create a reference camera from the metadata.
        ref_camera = camera_from_param(param, rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                                       perm=Permutation.NED)
        ref_camera_err = sad(camera_reprojection_errors(points, ref_camera))

        # Camera from the solved pose.
        camera0_err = sad(obj_f(points, t, np.array(ypr), fov))
        if camera0_err > ref_camera_err:
            print("Camera0 reprojection_error > ref camera for frame id: %s" %
                  image_id)

        if fov_error:
            fov_err = np.radians((param["horizontal-fov"] * 1.09,
                                  param["vertical-fov"] * .99))
            intrinsic, permute = intrinsic_and_permute(fov_err)
            ret, ypr, t = solve_pose_epnp(points, intrinsic, permute)

            camera1_err = sad(obj_f(points, t, np.array(ypr), fov_err))

            print("fov: %s" % np.degrees(fov))
            print("fov_err: %s" % np.degrees(fov_err))
            print("fov_min: %s" % np.degrees(
                optimize_fov(points, t, np.array(ypr), fov_err)))

    else:
        print("Failed to solve pose for frame id: %d" % image_id)


def run(path, confidence_limit=0.7, fov_error=False):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] >= confidence_limit:
            compare_image(image, fov_error)
