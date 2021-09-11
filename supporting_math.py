import cv2 as cv
import numpy as np

# Roate image about the center
# angle is in degrees
def rotate_image(image, angle, scale = 1):
    width = image.shape[1]
    height = image.shape[0]
    image_center = tuple(np.array((width, height)) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, scale)
    r = np.deg2rad(angle)
    new_width = scale*(abs(np.sin(r)*height) + abs(np.cos(r)*width))
    new_height = scale*(abs(np.sin(r)*width) + abs(np.cos(r)*height))
    # Translate center of image
    translate_x = (new_width - width)/2
    translate_y = (new_height - height)/2
    rot_mat[0,2] += translate_x
    rot_mat[1,2] += translate_y
    return(cv.warpAffine(image, rot_mat, dsize=(int(new_width), int(new_height))))

def adjust_image_loc_for_bottom_center_roation(x, y, height, angle, scale = 1):
    r = np.deg2rad(angle)
    x -= round(scale*height*np.sin(r)/2)
    y -= round(scale*height*np.cos(r)/2)
    return(x, y)

def wrap_orientation(angle):
      if angle < 0:
          angle += 360
      elif angle > 360:
          angle -= 360
      if angle < 0 or angle > 360:
          return wrap_orientation(angle)
      return angle

def line_points_low(x0, y0, x1, y1):
    fill = []
    dx = x1 - x0
    dy = y1 - y0
    y_inc = 1
    if dy < 0:
        dy *= -1
        y_inc = -1
    D = 2*dy - dx
    y = y0
    for x in range(x0,x1+1):
        fill.append([x,y])
        if D > 0:
            y += y_inc
            D -= 2*dx
        D += 2*dy
    return fill

def line_points_high(x0, y0, x1, y1):
    fill = []
    dx = x1 - x0
    dy = y1 - y0
    x_inc = 1
    if dx < 0:
        dx *= -1
        x_inc = -1
    D = 2*dx - dy
    x = x0
    for y in range(y0,y1+1):
        fill.append([x,y])
        if D > 0:
            x += x_inc
            D -= 2*dy
        D += 2*dx
    return fill

def get_line_points(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return line_points_low(x1, y1, x0, y0)
        else:
            return line_points_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return line_points_high(x1, y1, x0, y0)
        else:
            return line_points_high(x0, y0, x1, y1)


def test_bresenham_dx_g_dy():
    fill0 = get_line_points(0,1,6,4)
    fill1 = get_line_points(6,4,0,1)
    test_status = True
    for element in fill0:
        if element not in fill1:
            test_status = False
    if not test_status:
        print(fill0)
        print(fill1)
    else:
        print("pass")

def test_bresenham_dy_g_dx():
    fill0 = get_line_points(1,0,4,6)
    fill1 = get_line_points(4,6,1,0)
    test_status = True
    for element in fill0:
        if element not in fill1:
            test_status = False
    if not test_status:
        print(fill0)
        print(fill1)
    else:
        print("pass")

def draw_bresenham(x0, y0, x1, y1):
    fill = get_line_points(x0,y0,x1,y1)
    print(fill)
    for y in range(20):
        for x in range(20):
            if [x,y] in fill:
                print("1 ", end="")
            else:
                print("0 ", end="")
        print("")

if __name__ == "__main__":
    draw_bresenham(4,6,1,0)
    print()
    draw_bresenham(6,4,0,1)
    test_bresenham_dx_g_dy()
    test_bresenham_dy_g_dx()
