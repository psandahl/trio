import math
import unittest

import trio.common.math as common_math


class CommonMathTestCase(unittest.TestCase):
    def test_radians_to_degrees(self):
        # Empty lists or tuples.
        xs = []
        self.assertEqual(xs, common_math.degrees(xs))
        xs = ()
        self.assertEqual(xs, common_math.degrees(xs))

        # Simple angles.
        xs = [0.0, math.pi / 2.0, math.pi, math.pi * 1.5]
        self.assertAlmostEqual([0.0, 90.0, 180.0, 270.0],
                               common_math.degrees(xs))
        xs = (0.0, math.pi / 2.0, math.pi, math.pi * 1.5)
        self.assertAlmostEqual((0.0, 90.0, 180.0, 270.0),
                               common_math.degrees(xs))

        # Shall accept only lists or tuples.
        with self.assertRaises(TypeError):
            common_math.degrees(1)

    def test_degrees_to_radians(self):
        # Empty lists or tuples.
        xs = []
        self.assertEqual(xs, common_math.radians(xs))
        xs = ()
        self.assertEqual(xs, common_math.radians(xs))

        # Simple angles.
        xs = [0.0, 90.0, 180.0, 270.0]
        self.assertAlmostEqual([0.0, math.pi / 2.0, math.pi, math.pi * 1.5],
                               common_math.radians(xs))
        xs = (0.0, 90.0, 180.0, 270.0)
        self.assertAlmostEqual((0.0, math.pi / 2.0, math.pi, math.pi * 1.5),
                               common_math.radians(xs))

        # Shall accept only lists or tuples.
        with self.assertRaises(TypeError):
            common_math.degrees(1)
