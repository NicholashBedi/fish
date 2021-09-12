import plants
import numpy as np
import cv2 as cv

class TestForce(plants.Plants):
    def set_default_values(self):
        self.x = self.width//2
        self.y = self.height//2
        self.spring_const = 1
        self.damper_const = 1
        self.mass_inv = 1.0/1.0
        self.prev_angle = self.default_angle
        self.dangle = 0
        self.angle_velocity = 0
        self.dt = 0.1
        self.force_wind = 20

    def on_trackbar(self, val):
        pass

    def make_trackbars(self):
        self.trackbar_window = "trackbar"
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("Force_left", self.trackbar_window, 10, 100, self.on_trackbar)
        cv.createTrackbar("Mass", self.trackbar_window, 10, 100, self.on_trackbar)
        cv.createTrackbar("Spring Const", self.trackbar_window, 10, 100, self.on_trackbar)
        cv.createTrackbar("Damper Const", self.trackbar_window, 10, 100, self.on_trackbar)

    def get_trackbar_updates(self):
        self.force_wind = cv.getTrackbarPos("Force_left", self.trackbar_window)
        mass = cv.getTrackbarPos("Mass", self.trackbar_window)
        if mass == 0:
            mass = 0.1
        self.mass_inv = 1/mass

        self.spring_const = cv.getTrackbarPos("Spring Const", self.trackbar_window)
        self.damper_const = cv.getTrackbarPos("Damper Const", self.trackbar_window)

    def calc_acceleraion(self):
        return (self.force_wind*np.cos(np.deg2rad(self.dangle + self.default_angle)) \
                - self.spring_const*self.dangle \
                - self.damper_const*self.angle_velocity) * self.mass_inv

    def update(self):
        temp_angle_velocity = self.angle_velocity
        self.angle_velocity += self.dt * self.calc_acceleraion()
        self.dangle += self.dt * temp_angle_velocity

    def main(self):
        self.make_trackbars()
        self.set_default_values()

        while True:
            self.get_trackbar_updates()
            self.base_image = np.full([self.width,self.height, 4], 255, dtype=np.uint8)
            self.update()
            self.display_leaf(self.x, self.y,
                        np.rad2deg(self.default_angle + self.dangle), scale = 2)
            if cv.waitKey(10) & 0xFF == ord('q'):
                  break

p = TestForce()
p.main()
