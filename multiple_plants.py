import plants
import cv2 as cv
import numpy as np
import math
import supporting_math as sm
import wind

WIDTH = 600
HEIGHT = 600
n_plants = 100
plant_loc = np.random.randint(0, WIDTH, size=(n_plants, 2))

background_img = cv.imread("images/plant_background_test_1.png", cv.IMREAD_GRAYSCALE)
plant_loc = sm.get_samples_from_prob_image(background_img, 100)
background_img = cv.cvtColor(background_img, cv.COLOR_GRAY2BGR)

plant_def_angle = np.random.uniform(-10,10, size=(n_plants))
all_plants = []
for i in range(n_plants):
    all_plants.append(plants.Plants(i, plant_loc[i, 0], plant_loc[i, 1], plant_def_angle[i]))

base_image = np.full([WIDTH, HEIGHT, 4], 255, dtype=np.uint8)
base_image[:,:,:-1] = background_img
for p in all_plants:
    x, y, img = p.get_leaf_image()
    base_image = sm.insert_image(x,y,img,base_image)

w = wind.Wind()
w.make_trackbars()

t = 0
while(True):
    wind_angle = cv.getTrackbarPos("angle", w.trackbar_window)
    cv.imshow("Display window", base_image)
    base_image = np.full([WIDTH, HEIGHT, 4], 255, dtype=np.uint8)
    base_image[:,:,:-1] = background_img
    wind_force = w.get_wind(wind_angle, t)
    # wind_force = np.full([WIDTH, HEIGHT], 255, dtype=np.uint8)
    cv.imshow("Wind", wind_force)
    t += cv.getTrackbarPos("t-inc", w.trackbar_window)
    for p in all_plants:
        p.set_wind(wind_force)
        p.update()
        x, y, img = p.get_leaf_image()
        base_image = sm.insert_image(x,y,img,base_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
          break
