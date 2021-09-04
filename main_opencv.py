import cv2 as cv
import sys
import math
import numpy as np

WIDTH = 512
HEIGHT = 512

def draw_fish(base_image, fish_img, position, velocity):
    angle = math.atan2(velocity[1], velocity[0]) * 180/ math.pi
    rot_fish = rotate_image(fish_img, angle)
    base_image[position[1] - math.floor(rot_fish.shape[0]/2):
               position[1] + math.ceil( rot_fish.shape[0]/2),
               position[0] - math.floor(rot_fish.shape[1]/2):
               position[0] + math.ceil( rot_fish.shape[1]/2)] = rot_fish
    base_image[base_image[:,:,3] == 0] = [255,255,255,255]
    return base_image

def rotate_image(image, angle, scale = 0.25):
    initial_height, initial_width = image.shape[:2]
    image_center = tuple(np.array((initial_width, initial_height)) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, scale)
    r = np.deg2rad(angle)
    new_width = scale*(abs(np.sin(r)*initial_height)
            + abs(np.cos(r)*initial_width))
    new_height = scale*(abs(np.sin(r)*initial_width)
            + abs(np.cos(r)*initial_height))
    # Translate center of image
    translate_x = (new_width-initial_width)/2
    translate_y = (new_height-initial_height)/2
    rot_mat[0,2] += translate_x
    rot_mat[1,2] += translate_y
    result = cv.warpAffine(image, rot_mat, dsize=(int(new_width), int(new_height)))
    return result

def on_trackbar(x):
    x_offset = cv.getTrackbarPos('x_position', trackbar_window)
    y_offset = cv.getTrackbarPos('y_position', trackbar_window)
    angle = cv.getTrackbarPos('angle', trackbar_window)
    rot_fish = rotate_image(fish_img, angle)
    base_image = np.full([WIDTH,HEIGHT, 4], 255, dtype=np.uint8)
    base_image[:,:,3] = 0
    base_image[y_offset - math.floor(rot_fish.shape[0]/2):
               y_offset + math.ceil(rot_fish.shape[0]/2),
               x_offset - math.floor(rot_fish.shape[1]/2):
               x_offset + math.ceil(rot_fish.shape[1]/2)] = rot_fish
    trans_mask = base_image[:,:,3] == 0
    base_image[trans_mask] = [255,255,255,255]
    cv.imshow("Display window", base_image)

fish_img = cv.imread("fish.png", cv.IMREAD_UNCHANGED)
np.set_printoptions(threshold=sys.maxsize)
# fish[:,:,3] = 255
# print(fish)
trackbar_window = "Window name"
cv.namedWindow(trackbar_window)
cv.createTrackbar("x_position", trackbar_window, math.ceil(fish.shape[1]/2),
                 WIDTH - math.ceil(fish.shape[1]/2), on_trackbar)
cv.createTrackbar("y_position", trackbar_window, math.ceil(fish.shape[0]/2),
                HEIGHT - math.ceil(fish.shape[0]/2), on_trackbar)
cv.createTrackbar("angle", trackbar_window, 0, 360, on_trackbar)
on_trackbar(0)
k = cv.waitKey(0)
