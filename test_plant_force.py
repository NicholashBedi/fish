import plants
import numpy as np
import cv2 as cv

class TestForce(plants.Plants):
    def set_test_force_values(self):
        self.x = self.width//2
        self.y = self.height//2

    def on_trackbar(self, val):
        pass

    def make_trackbars(self):
        self.trackbar_window = "trackbar"
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("Force_left", self.trackbar_window, 13, 100, self.on_trackbar)
        cv.createTrackbar("Force_right", self.trackbar_window, 0, 100, self.on_trackbar)
        cv.createTrackbar("Mass", self.trackbar_window, 10, 100, self.on_trackbar)
        cv.createTrackbar("Spring Const", self.trackbar_window, 10, 100, self.on_trackbar)
        cv.createTrackbar("Damper Const", self.trackbar_window, 15, 100, self.on_trackbar)

    def get_trackbar_updates(self):
        self.force_wind = cv.getTrackbarPos("Force_left", self.trackbar_window) \
                          - cv.getTrackbarPos("Force_right", self.trackbar_window)
        mass = cv.getTrackbarPos("Mass", self.trackbar_window)
        if mass == 0:
            mass = 0.1
        self.mass_inv = 1/mass
        self.spring_const = cv.getTrackbarPos("Spring Const", self.trackbar_window)
        self.damper_const = cv.getTrackbarPos("Damper Const", self.trackbar_window)

    def main(self):
        self.make_trackbars()
        self.set_test_force_values()

        while True:
            self.get_trackbar_updates()
            self.base_image = np.full([self.width,self.height, 4], 255, dtype=np.uint8)
            self.update()
            self.display_leaf_at_loc(self.x, self.y, scale = 2)
            if cv.waitKey(10) & 0xFF == ord('q'):
                  break

p = TestForce()
p.main()
