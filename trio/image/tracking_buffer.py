import numpy as np
import cv2 as cv
import math


class TrackingBuffer:

    orig_images = []
    gray_images = []
    flow_images = []
    cameras = []
    width = 20  # Must be at least 2

    def __init__(self):
        return

    def add_image(self, image, camera):
        # Clean oldest data.
        if len(self.orig_images) == self.width:
            self.orig_images.pop(0)
            self.gray_images.pop(0)
            self.flow_images.pop(0)
            self.cameras.pop(0)

        # Gray convert the image.
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        self.orig_images.append(image)
        self.gray_images.append(gray)
        self.cameras.append(camera)

        gs = len(self.gray_images)
        if gs >= 2:
            flow = cv.calcOpticalFlowFarneback(self.gray_images[gs - 2],
                                               self.gray_images[gs - 1],
                                               None, 0.5, 3, 15, 3, 5, 1.2, 0)
            self.flow_images.append(flow)

    def has_content(self):
        return len(self.orig_images) > 0

    def oldest_image(self):
        return self.orig_images[0]

    def newest_image(self):
        return self.orig_images[len(self.orig_images) - 1]

    def oldest_camera(self):
        return self.cameras[0]

    def newest_camera(self):
        return self.cameras[len(self.cameras) - 1]

    def track(self, px):
        if len(self.flow_images) == 0:
            return px

        uv = np.array(px)

        #print("Start tracking - input u=%.5f, v=%.5f" % (uv[0], uv[1]))
        for flow in self.flow_images:
            # Integer and fractional parts of uv.
            iu = math.floor(uv[0])
            iv = math.floor(uv[1])

            if iu < flow.shape[1] and iv < flow.shape[0]:
                wu = uv[0] - iu
                wv = uv[1] - iv

                # Get interpolation weights.
                w00 = (1.0 - wu) * (1.0 - wv)
                w10 = (1.0 - wu) * wv
                w01 = wu * (1.0 - wv)
                w11 = wu * wv

                # print("Interp. weights - w00=%.3f, w10=%.3f, w01=%.3f, w11=%.3f" %
                #      (w00, w10, w01, w11))

                weighted_change = flow[iv, iu] * w00 + flow[iv + 1, iu] * w10 \
                    + flow[iv, iu + 1] * w01 + flow[iv + 1, iu + 1] * w11

                # print("Weighted change - u=%.5f, v=%.5f" %
                #      (weighted_change[0], weighted_change[1]))

                uv += weighted_change
                #print("Iteration update - u=%.5f, v=%.5f" % (uv[0], uv[1]))

        #print("Final update - u=%.5f, v=%.5f" % (uv[0], uv[1]))

        return (uv[0], uv[1])
