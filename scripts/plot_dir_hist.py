import numpy as np
import matplotlib.pyplot as plt


random_angles = (2 * np.pi * np.random.rand(200)) - np.pi

random_angles2 = random_angles + 2*np.pi
random_angles3 = np.remainder(random_angles2, 2 * np.pi)

N = 8
N_ = N*2
ranges = np.linspace(0, 2 * np.pi + np.pi/N_, N_ + 1, endpoint=False)

dir_hist = np.histogram(random_angles, ranges)[0]
first_elems  = dir_hist[0::2]
second_elems = np.roll(dir_hist, 1)[0::2]
dir_hist_end = first_elems + second_elems
# Compute pie slices
radii = dir_hist[0]
width = np.pi / 4
colors = np.random.rand(N,3)
theta2 = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

ax = plt.subplot(111, projection='polar')
ax.bar(theta2, dir_hist_end, width=width, bottom=0.0, color=colors, alpha=0.5)

plt.show()