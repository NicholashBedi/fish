import cv2 as cv
import numpy as np

background_img = cv.imread("images/plant_background_test_1.png", cv.IMREAD_GRAYSCALE)
plant_loc_prob = 1 - background_img/255
plant_loc_prob = plant_loc_prob.flatten()
plant_loc_prob /= np.sum(plant_loc_prob)

plant_loc = np.random.choice(600*600, 100, replace = False, p = plant_loc_prob)

background_img = cv.cvtColor(background_img, cv.COLOR_GRAY2BGR)

for loc in plant_loc:
    cv.circle(background_img, (loc%600, loc//600), 2, (0,0,255))
cv.imshow("Plants", background_img)

while(True):
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
