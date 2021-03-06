import numpy as np
import cv2 as cv
import math
import supporting_math as sm
import wind_function as wf

class Wind:
    def __init__(self):
        self.debug = True
        self.width = 600
        self.height = 600

    def on_trackbar(self, val):
        pass

    def make_trackbars(self):
        self.trackbar_window = "trackbar"
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("angle", self.trackbar_window, 0, 360, self.on_trackbar)
        cv.createTrackbar("t-inc", self.trackbar_window, 0, 100, self.on_trackbar)
    def wind_force(self, x, t):
        # return wf.zero(x, self.width, t)
        # return wf.square_wave(x, self.width, t)
        # return wf.sin_wave(x, self.width, t)
        return wf.perlin(x, self.width, t)

    def get_wind(self, degrees, t):
        larger_length = math.ceil(max(self.height, self.width)*math.sqrt(2))
        larger_sin_image = np.zeros([larger_length, larger_length], dtype=np.uint8)
        for x in range(larger_length):
            larger_sin_image[:,x] = self.wind_force(x, t)
        rotated_image = sm.rotate_image(larger_sin_image, degrees)
        r_center_h = rotated_image.shape[0]//2
        r_center_w = rotated_image.shape[1]//2
        return rotated_image[r_center_h-self.height//2:r_center_h+self.height//2,
                                        r_center_w-self.width//2:r_center_w+self.width//2]
    def display_wind(self, t):
        angle = cv.getTrackbarPos("angle", self.trackbar_window)
        cv.imshow("display", self.get_wind(angle, t))

if __name__ == "__main__":
    w = Wind()
    w.make_trackbars()
    t = 0
    while True:
        w.display_wind(t)
        t += cv.getTrackbarPos("t-inc", w.trackbar_window)
        if cv.waitKey(1) & 0xFF == ord('q'):
          break
