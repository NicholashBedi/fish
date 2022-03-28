import numpy as np
import math
import perlin_noise_1d


# Wind force is between -1 and 1 where -1 is max force left
# and 1 is max force right
# For Image 0 (Black) is max left and 255 (white) is max right
def wind_force_to_img(force):
    return round((force + 1)*255/2)

def img_to_force(img_val):
    return (img_val*2 / 255) - 1

def sin_wave(x, width, t):
    return wind_force_to_img(np.sin(2*math.pi*(x+t)/width))

def square_wave(x, width, t):
    if (x+t) % width > width/2:
        return wind_force_to_img(1)
    else:
        return wind_force_to_img(-1)

def zero(x, width, t):
    return wind_force_to_img(0)
def perlin(x, width, t):
    return(wind_force_to_img(perlin_noise_1d.perlin( (x+t)/400)))
