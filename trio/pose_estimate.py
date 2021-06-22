from scipy import linalg
import numpy as np

import json
import math

from .common.camera import Camera, Permutation, \
    camera_from_param, camera_reprojection_errors
from .common.math import sad
from .common.matrix import matrix_permute_ned, matrix_intrinsic
from .common.linear import solve_pose_epnp


def obj_from_file(path):
    return json.load(open(path))


def intrinsic_and_permute(param):
    intrinsic = matrix_intrinsic(np.radians((param["horizontal-fov"],
                                             param["vertical-fov"])),
                                 rect=np.array([-0.5, -0.5, 1.0, 1.0]))
    return (intrinsic, matrix_permute_ned())


def close_angles(theta0, theta1):
    return abs(theta0 - theta1) < 0.00001  # Close enough?


def equal_angles(yaw0, yaw1, pitch0, pitch1, roll0, roll1):
    return close_angles(yaw0, yaw1) and close_angles(pitch0, pitch1) and \
        close_angles(roll0, roll1)


def position_distance(pos0, param):
    return linalg.norm(pos0 - np.array((param["x"], param["y"], param["z"])))


def compare_image(image):
    image_id = image["image-id"]
    param = image["camera-parameters"]
    points = image["point-correspondences"]

    # Start by solving for pose.
    intrinsic, permute = intrinsic_and_permute(param)
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

        # Create a reference camera from the metadata.
        ref_camera = camera_from_param(param, rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                                       perm=Permutation.NED)
        ref_camera_err = sad(camera_reprojection_errors(points, ref_camera))

        # Camera from the solved pose.
        camera0 = Camera(t, np.array(ypr),
                         np.radians((param["horizontal-fov"],
                                     param["vertical-fov"])),
                         rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                         perm=Permutation.NED)
        camera0_err = sad(camera_reprojection_errors(points, camera0))
        if camera0_err > ref_camera_err:
            print("Camera0 reprojection_error > ref camera for frame id: %s" %
                  image_id)
    else:
        print("Failed to solve pose for frame id: %d" % image_id)


def run(path, confidence_limit=0.7):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] >= confidence_limit:
            compare_image(image)
