import numpy as np
import cv2 as cv

from .utils import remove_symbols


class MatchingBuffer:
    width = 0
    content_buffer = []

    def __init__(self, width):
        self.width = width

    def push(self, image, camera, points, confidence):
        if self.width == len(self.content_buffer):
            self.content_buffer.pop(0)

        entry = dict()
        entry["orig-image"] = image
        entry["clean-image"] = remove_symbols(
            cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        entry["camera"] = camera
        entry["points"] = points
        entry["confidence"] = confidence

        self.content_buffer.append(entry)

    def has_valid_pairing(self):
        if self.width == len(self.content_buffer):
            e0 = self.content_buffer[0]
            e1 = self.content_buffer[self.width - 1]
            return e0["confidence"] > 0.75 and e1["confidence"] > 0.75
        else:
            return False

    def valid_pairing(self):
        return (self.content_buffer[0], self.content_buffer[self.width - 1])
