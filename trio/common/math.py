import math


def degrees(xs):
    if type(xs) == list:
        return list(map(math.degrees, xs))
    elif type(xs) == tuple:
        return tuple(map(math.degrees, xs))
    else:
        raise TypeError("xs must be either list or tuple")


def radians(xs):
    if type(xs) == list:
        return list(map(math.radians, xs))
    elif type(xs) == tuple:
        return tuple(map(math.radians, xs))
    else:
        raise TypeError("xs must be either list or tuple")
