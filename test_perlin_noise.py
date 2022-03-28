import matplotlib.pyplot as plt
import numpy as np
import perlin_noise_1d


x = np.linspace(0, 9, 1000)
# x = [2.1]
y = []
for x_i in x:
    y.append(perlin_noise_1d.perlin(x_i))

plt.plot(x,y)
plt.show()
