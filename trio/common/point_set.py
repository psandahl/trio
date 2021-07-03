import numpy as np
import math


class PointSet:
    points = []
    dist = 0.0

    def __init__(self, dist):
        self.points = []
        self.dist = math.sqrt(dist)

    def add(self, p):
        if not self.contains(p):
            self.points.append(p)

    def contains(self, p):
        for point in self.points:
            x, y, z = p - point
            dist = x**2 + y**2 + z**2
            if dist < self.dist:
                return True

        return False
