import cv2 as cv
import numpy as np
import supporting_math as sm

background_img = cv.imread("images/plant_background_test_1.png", cv.IMREAD_GRAYSCALE)
plant_loc = sm.get_samples_from_prob_image(background_img, 100)

background_img = cv.cvtColor(background_img, cv.COLOR_GRAY2BGR)

for loc in plant_loc:
    cv.circle(background_img, (loc[0], loc[1]), 2, (0,0,255))
cv.imshow("Plants", background_img)

while(True):
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
