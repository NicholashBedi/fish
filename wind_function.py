import numpy as np
import math

def sin_wave(x, width, t):
    return 255/2*(np.sin((x+width/2)/width*2*math.pi + t) + 1)
