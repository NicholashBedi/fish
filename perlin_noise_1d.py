import numpy as np
import math

random_vals = np.random.uniform(size=10)
random_vals[-1] = random_vals[0]

def lerp(y0, y1, a):
    return (1-a)*y0 + a*y1
def cos_func(a):
    return((1-math.cos(math.pi*a))/2)
def cerp(y0, y1, a):
    return y0*(1-cos_func(a)) + y1*cos_func(a)

def noise(x_i):
    x_floor = math.floor(x_i)
    alpha = x_i - x_floor
    if x_floor+1 >= len(random_vals):
        return random_vals[-1]
    return(cerp(random_vals[x_floor], random_vals[x_floor+1], alpha))

def perlin(x, p = 0.5, octaves = 6):
    y = 0
    for i in range(octaves):
        y += p**(i)*noise((2**i*x)%9)
    return(y - 1)
