import numpy as np
import cv2 as cv

from .utils import remove_symbols


class MatchingBuffer:
    width = 0
    content_buffer = []
    detector = None
    matcher = None

    def __init__(self, width):
        self.width = width
        self.detector = cv.AKAZE_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def push(self, image, camera, points, confidence):
        if self.width == len(self.content_buffer):
            self.content_buffer.pop(0)

        clean_image = remove_symbols(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        kpt, desc = self.detector.detectAndCompute(clean_image, None)

        entry = dict()
        entry["orig-image"] = image
        entry["clean-image"] = clean_image
        entry["camera"] = camera
        entry["points"] = points
        entry["confidence"] = confidence
        entry["keypoints"] = kpt
        entry["descriptors"] = desc

        self.content_buffer.append(entry)

    def has_valid_pairing(self):
        if self.width == len(self.content_buffer):
            entry0 = self.content_buffer[0]
            entry1 = self.content_buffer[self.width - 1]
            return entry0["confidence"] > 0.7 and entry1["confidence"] > 0.7
        else:
            return False

    def valid_pairing(self):
        entry0 = self.content_buffer[0]
        entry1 = self.content_buffer[self.width - 1]

        matches = self.matcher.match(
            entry0["descriptors"], entry1["descriptors"])
        matches = sorted(matches, key=lambda x: x.distance)

        return (entry0, entry1, matches)
