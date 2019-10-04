#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np


# dataset = 'NGSIM'
dataset = 'ARGO'

plt.ion()
if dataset == 'ARGO':
    map_range = np.array([[2570, 2600, 1180, 1210], [2640, 2670, 1240, 1270]])

    with open("data_sample/ARGO_final_DPGP_train4_alpha_1", "rb") as np:
        mixture_models = pickle.load(np)
        load_frames = mixture_models.frames
elif dataset == 'NGSIM':
    with open("frame_US_101", "rb") as np:  # load saved frames
        load_frames = pickle.load(np)

# for i in range(10):
for i in range(len(load_frames)):
# the on road is much more stable, however the off road ones are quite noisy
# for i in range(1000):
    plt.cla()
    frame_temp = load_frames[i]
    plt.quiver(frame_temp.x, frame_temp.y, frame_temp.vx, frame_temp.vy)
    if dataset == 'ARGO':
        plt.xlim([2570, 2600])
        plt.ylim([1180, 1210])
    elif dataset == 'NGSIM':
        plt.xlim([0, 60])
        plt.ylim([1300, 1600])
    plt.show()
    plt.pause(0.05)


plt.ioff()