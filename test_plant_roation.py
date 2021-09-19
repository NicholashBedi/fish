import plants
import numpy as np
import cv2 as cv

class TestRotation(plants.Plants):
    def on_trackbar(self, val):
        pass
    def make_trackbars(self):
        self.trackbar_window = "trackbar"
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("x", self.trackbar_window, 300, self.width - 60, self.on_trackbar)
        cv.createTrackbar("y", self.trackbar_window, 300, self.height - 60, self.on_trackbar)
        cv.createTrackbar("angle", self.trackbar_window, 0, 360, self.on_trackbar)

    def test(self):
        self.make_trackbars()
        while True:
            self.base_image = np.full([self.width,self.height, 4], 255, dtype=np.uint8)
            x = cv.getTrackbarPos("x", self.trackbar_window)
            y = cv.getTrackbarPos("y", self.trackbar_window)
            angle = cv.getTrackbarPos("angle", self.trackbar_window)
            self.display_lead_at_loc_with_ang(x,y,angle, 5)
            if cv.waitKey(1) & 0xFF == ord('q'):
                  break

p = TestRotation()
p.test()
