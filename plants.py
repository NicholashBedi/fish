import numpy as np
import cv2 as cv
import math
import supporting_math as sm
import sys

class Plants:
    def __init__(self, id_i = 0, x_i = 10, y_i = 10, angle_i = 0, debug_i = False):
        self.id = id_i
        self.debug = debug_i
        self.width = 600
        self.height = 600
        self.x = x_i
        self.y = y_i
        # Angle from verticle in radians
        self.default_angle = np.deg2rad(angle_i)
        self.base_image = np.full([self.width,self.height, 4], 255, dtype=np.uint8)
        self.set_default_force_values()
        self.get_plant_img()

    def set_default_force_values(self):
        self.spring_const = 10
        self.damper_const = 15
        self.mass_inv = 1.0/10.0
        self.dangle = 0
        self.angle_velocity = 0
        self.dt = 0.1
        self.force_wind = 0

    def set_wind(self, wind):
        self.force_wind = (wind[self.y, self.x] - 255/2)/10

    def display_force_values(self):
        wind = self.force_wind*np.cos(self.dangle + self.default_angle)
        spring = -1*self.spring_const*self.dangle
        damper = -1*self.damper_const*self.angle_velocity
        total = (wind + spring + damper)*self.mass_inv
        font = cv.FONT_HERSHEY_SIMPLEX
        colour = (255, 0, 0)
        ID_str = "ID: {:3d}".format(self.id)
        total_str = "Total: {:5.3f}".format(total)
        wind_str = "Wind: {:5.3f}".format(wind)
        spring_str = "Spring: {:5.3f}".format(spring)
        damper_str = "Damper: {:5.3f}".format(damper)
        angle = "Angle: {:5.3f}".format(self.dangle + self.default_angle)
        to_print = [ID_str, wind_str, spring_str, damper_str, total_str, angle]
        return_str = ""
        for i, print_str in enumerate(to_print):
            return_str += " " + print_str
            # self.base_image = cv.putText(self.base_image, print_str,
            #                 (300, (i+1)*30), font, 1, (255,0,0), 1, cv.LINE_AA)
        print(return_str)

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
        return self.plant_img

    def display_leaf(self, scale = 0.5):
        self.display_leaf_at_loc(self.x, self.y, self.dangle + self.default_angle, scale)

    # x increases as you go right
    # y increases as you go down
    # (0,0) is at top left
    # angle of plant, 0 is stright up, as angle increases it moves counter clockwise
    # angle in degrees
    def display_leaf_at_loc(self, x, y, scale = 0.5):
        self.x = x
        self.y = y
        x_adjust, y_adjust, rot_image = self.get_leaf_image(scale)
        self.base_image = sm.insert_image(x_adjust, y_adjust,
                                        rot_image, self.base_image)
        cv.imshow("Display window", self.base_image)

    def get_leaf_image(self, scale = 0.5):
        angle = np.rad2deg(self.dangle + self.default_angle)
        rotated_image = sm.rotate_image(self.plant_img, angle, scale)
        x_adjust, y_adjust = sm.adjust_image_loc_for_bottom_center_roation(
                                self.x, self.y, self.plant_img.shape[0], angle, scale)
        return (x_adjust, y_adjust, rotated_image)


if __name__ == "__main__":
    pass
