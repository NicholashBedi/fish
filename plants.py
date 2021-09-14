import numpy as np
import cv2 as cv
import math
import supporting_math as sm
import sys

class Plants:
    def __init__(self):
        self.debug = True
        self.width = 600
        self.height = 600
        # Angle from verticle in radians
        self.default_angle = np.deg2rad(-30)
        self.set_default_force_values()
        self.get_plant_img()

    def set_default_force_values(self):
        self.spring_const = 1
        self.damper_const = 1
        self.mass_inv = 1.0/1.0
        self.dangle = 0
        self.angle_velocity = 0
        self.dt = 0.1
        self.force_wind = 20

    def display_force_values(self):
        wind = self.force_wind*np.cos(self.dangle + self.default_angle)
        spring = -1*self.spring_const*self.dangle
        damper = -1*self.damper_const*self.angle_velocity
        total = (wind + spring + damper)*self.mass_inv
        font = cv.FONT_HERSHEY_SIMPLEX
        colour = (255, 0, 0)
        total_str = "Total : {:5.3f}".format(total)
        wind_str = "Wind  : {:5.3f}".format(wind)
        spring_str = "Spring: {:5.3f}".format(spring)
        damper_str = "Damper: {:5.3f}".format(damper)
        angle = "Angle: {:5.3f}".format(self.dangle + self.default_angle)
        to_print = [wind_str, spring_str, damper_str, total_str, angle]
        for i, print_str in enumerate(to_print):
            self.base_image = cv.putText(self.base_image, print_str,
                            (300, (i+1)*30), font, 1, (255,0,0), 1, cv.LINE_AA)

    def calc_acceleraion(self):
        if self.debug:
            self.display_force_values()
        return (self.force_wind*np.cos(self.dangle + self.default_angle) \
                - self.spring_const*self.dangle \
                - self.damper_const*self.angle_velocity) * self.mass_inv

    def update(self):
        temp_angle_velocity = self.angle_velocity
        self.angle_velocity += self.dt * self.calc_acceleraion()
        self.dangle += self.dt * temp_angle_velocity

    def get_plant_img(self):
        temp = cv.imread("plant_images/leaf.png")
        self.plant_img = np.zeros((temp.shape[0],temp.shape[1], 4))
        # set RGB of image
        self.plant_img[:, :, :-1] = temp
        # Set remaining transparency
        non_transparent = self.plant_img[:, :, 0] == 0
        self.plant_img[non_transparent, 3] = 255

    # x increases as you go right
    # y increases as you go down
    # (0,0) is at top left
    # angle of plant, 0 is stright up, as angle increases it moves counter clockwise
    # angle in degrees
    def display_leaf(self, x, y, angle, scale = 0.5):
        rotated_image = sm.rotate_image(self.plant_img, angle, scale)
        x_adjust, y_adjust = sm.adjust_image_loc_for_bottom_center_roation(
                                x, y, self.plant_img.shape[0], angle, scale)
        self.base_image = sm.insert_image(x_adjust, y_adjust,
                                        rotated_image, self.base_image)
        cv.imshow("Display window", self.base_image)


if __name__ == "__main__":
    pass
