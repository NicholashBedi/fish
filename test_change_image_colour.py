import cv2 as cv
import supporting_math as sm

def on_trackbar(x):
    pass

fish_img = cv.imread("images/fish.png", cv.IMREAD_UNCHANGED)
trackbar_window = "colour"

cv.namedWindow(trackbar_window)
cv.createTrackbar("blue", trackbar_window, 255, 255, on_trackbar)
cv.createTrackbar("green", trackbar_window, 0, 255, on_trackbar)
cv.createTrackbar("red", trackbar_window, 0, 255, on_trackbar)

while(True):
    b = cv.getTrackbarPos("blue", trackbar_window)
    g = cv.getTrackbarPos("green", trackbar_window)
    r = cv.getTrackbarPos("red", trackbar_window)
    fish_img = sm.change_colour(fish_img, b, g, r)
    cv.imshow("Display window", fish_img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
