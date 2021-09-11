import numpy as np
import cv2 as cv
import math
import supporting_math as sm
import sys

WIDTH = 600
HEIGHT = 600

class Plants:
    def __init__(self):
        temp = cv.imread("plant_images/leaf.png")
        self.plant_img = np.zeros((temp.shape[0],temp.shape[1], 4))
        self.plant_img[:, :, :-1] = temp
        non_transparent = self.plant_img[:, :, 0] == 0
        self.plant_img[non_transparent, 3] = 255
        self.trackbar_window = "trackbar"
        self.make_trackbars()

    def make_trackbars(self):
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("x", self.trackbar_window, 300, WIDTH - 60, self.on_trackbar)
        cv.createTrackbar("y", self.trackbar_window, 300, HEIGHT - 60, self.on_trackbar)
        cv.createTrackbar("angle", self.trackbar_window, 0, 360, self.on_trackbar)

    def on_trackbar(self, val):
        pass

    def display_leaf(self):
        self.base_image = np.full([WIDTH,HEIGHT, 4], 255, dtype=np.uint8)
        x = cv.getTrackbarPos("x", self.trackbar_window)
        y = cv.getTrackbarPos("y", self.trackbar_window)
        angle = cv.getTrackbarPos("angle", self.trackbar_window)
        rotated_image = sm.rotate_image(self.plant_img, angle, scale = 0.5)
        x_adjust, y_adjust = sm.adjust_image_loc_for_bottom_center_roation(
                                x, y, self.plant_img.shape[0], angle, scale = 0.5)
        self.base_image = sm.insert_image(x_adjust, y_adjust,
                                        rotated_image, self.base_image)
        cv.imshow("Display window", self.base_image)

p = Plants()
while True:
    p.display_leaf()
    if cv.waitKey(1) & 0xFF == ord('q'):
          break
