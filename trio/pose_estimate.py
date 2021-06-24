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


def angles_distance(yaw0, yaw1, pitch0, pitch1, roll0, roll1):
    return (yaw1 - yaw0, pitch1 - pitch0, roll1 - roll0)


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


def optimize_position(points, position, orientation, fov):
    obj = functools.partial(obj_f, points,
                            orientation=orientation, fov=fov)
    res = optimize.least_squares(
        lambda x: obj(position=x), position, method='lm')
    return res.x


def optimize_orientation(points, position, orientation, fov):
    obj = functools.partial(obj_f, points, position=position, fov=fov)

    res = optimize.least_squares(
        lambda x: obj(orientation=x), orientation, method='lm')
    return res.x


def ok_str(cond):
    if cond:
        return "OK"
    else:
        return "NOT OK"


def compare_image(image, fov_error):
    fov_err_scale = 1.2

    image_id = image["image-id"]
    param = image["camera-parameters"]
    points = image["point-correspondences"]

    fov = np.radians((param["horizontal-fov"], param["vertical-fov"]))
    if fov_error:
        fov *= fov_err_scale

    # Start by solving for pose.
    intrinsic, permute = intrinsic_and_permute(fov)
    ret, ypr, t = solve_pose_epnp(points, intrinsic, permute)
    if ret:
        print("===")
        print("Processing frame id: %d - confidence %.3f" %
              (image_id, image["confidence"]))

        # Convert solved rotation to degrees.
        yaw, pitch, roll = (math.degrees(ypr[0]),
                            math.degrees(ypr[1]),
                            math.degrees(ypr[2]))

        yaw_diff, pitch_diff, roll_diff = angles_distance(yaw, param["yaw"],
                                                          pitch, param["pitch"],
                                                          roll, param["roll"])
        print(" Differences in degrees yaw/pitch/roll: %.6f/%.6f/%.6f - %s" %
              (yaw_diff, pitch_diff, roll_diff,
               ok_str(equal_angles(yaw, param["yaw"], pitch, param["pitch"],
                                   roll, param["roll"]))
               ))

        print(" Difference in position (m): %.6f - %s" %
              (position_distance(t, param),
               ok_str(position_distance(t, param) < 1.0)))

        # Create a reference camera from the metadata.
        ref_camera = camera_from_param(param, rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                                       perm=Permutation.NED)
        ref_camera_err = sad(camera_reprojection_errors(points, ref_camera))

        # Camera from the solved pose.
        camera0_err = sad(obj_f(points, t, np.array(ypr), fov))

        print(" Reference camera error (SAD): %.12f" % ref_camera_err)
        print(" Pose reconstructed camera error (SAD): %.12f" % camera0_err)

        if fov_error:
            # Try to search for better fov - but use position from metadata.
            print(" Try solve stuff to compensate for incorrect fov")

            position = np.array((param["x"], param["y"], param["z"]))
            opt_fov = np.degrees(optimize_fov(points, position,
                                              np.array(ypr), fov))

            print("  Adjust horizontal fov from %.5f to %.5f (should be %.5f)" %
                  (np.degrees(fov[0]), opt_fov[0], param["horizontal-fov"]))
            print("  Adjust vertical fov from %.5f to %.5f (should be %.5f)" %
                  (np.degrees(fov[1]), opt_fov[1], param["vertical-fov"]))

            # With new fov, try search for better position.
            opt_fov = np.radians(opt_fov)
            opt_position = optimize_position(points, t, np.array(ypr), opt_fov)
            print("  Adjust position err from distance %.5fm to %.5fm" %
                  (linalg.norm(t - position), linalg.norm(opt_position - position)))

            opt_orientation = optimize_orientation(
                points, opt_position, ypr, opt_fov)
            yaw, pitch, roll = (math.degrees(opt_orientation[0]),
                                math.degrees(opt_orientation[1]),
                                math.degrees(opt_orientation[2]))

            yaw_diff, pitch_diff, roll_diff = angles_distance(yaw, param["yaw"],
                                                              pitch, param["pitch"],
                                                              roll, param["roll"])
            print(" Differences in degrees after opt yaw/pitch/roll: %.6f/%.6f/%.6f - %s" %
                  (yaw_diff, pitch_diff, roll_diff,
                   ok_str(equal_angles(yaw, param["yaw"], pitch, param["pitch"],
                                       roll, param["roll"]))
                   ))

    else:
        print("Failed to solve pose for frame id: %d" % image_id)


def run(path, confidence_limit=0.7, fov_error=False):
    images = obj_from_file(path)["images"]
    for image in images:
        if image["confidence"] >= confidence_limit:
            compare_image(image, fov_error)
