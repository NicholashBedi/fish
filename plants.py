import numpy as np
import cv2 as cv
import math
import supporting_math as sm
import sys

class Plants:
    def __init__(self):
        self.width = 600
        self.height = 600
        self.get_plant_img()

    def get_plant_img(self):
        temp = cv.imread("plant_images/leaf.png")
        self.plant_img = np.zeros((temp.shape[0],temp.shape[1], 4))
        # set RGB of image
        self.plant_img[:, :, :-1] = temp
        # Set remaining transparency
        non_transparent = self.plant_img[:, :, 0] == 0
        self.plant_img[non_transparent, 3] = 255

    def display_leaf(self, x, y, angle):
        rotated_image = sm.rotate_image(self.plant_img, angle, scale = 0.5)
        x_adjust, y_adjust = sm.adjust_image_loc_for_bottom_center_roation(
                                x, y, self.plant_img.shape[0], angle, scale = 0.5)
        self.base_image = sm.insert_image(x_adjust, y_adjust,
                                        rotated_image, self.base_image)
        cv.imshow("Display window", self.base_image)


if __name__ == "__main__":
    pass
