#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from mixtureModel import MixtureModel
import time
import numpy as np
from pattern_vis import *
import pickle

start_time = time.time()
dataset = 'ARGO'

map_range = np.array([[2570, 2600, 1180, 1210],
                      [2600, 2640, 1210, 1250],
                      [2640, 2670, 1240, 1270],
                      [2670, 2710, 1270, 1310],
                      [2710, 2740, 1300, 1330],
                      [2740, 2780, 1330, 1370],
                      [2780, 2810, 1360, 1390]])

train_flag = False
# train the model for each map cut
if train_flag:
    for i in range(map_range.shape[0]):
        area_idx = i
        with open("data_sample/argo_%d_%d_%d_%d" % (
                    map_range[area_idx, 0], map_range[area_idx, 1],
                    map_range[area_idx, 2], map_range[area_idx, 3]), "rb") as fp:
            load_frames = pickle.load(fp)

        a = MixtureModel(load_frames)
        a.mixture_model()

        for iter in range(3): # Gibbs Sampling Iterations
            # update each frame's indicator/assignment
            a.update_all_assignment()
            # update gaussian process patterns
            a.update_all_pattern()
            # show number of mixtures
            a.show_mixture_model()

        with open("data_sample/argo_MixtureModel_%d_%d_%d_%d" % (
                    map_range[area_idx, 0], map_range[area_idx, 1],
                    map_range[area_idx, 2], map_range[area_idx, 3]), "wb") as fb:
            pickle.dump(a, fb)

        print("---------------------------------------")
        print('DPGP finished!!')
        print('frame_idx: %d' % area_idx)
        print("--------------- %s ----------------" % (time.time() - start_time))

# visualize all the patterns and save to fig folder
pattern_vis_flag = True
if pattern_vis_flag:
    for j in range(map_range.shape[0]):
        area_idx = j
        WX, WY, ux_pos, uy_pos = velocity_field_visualization(map_range[area_idx, 0], map_range[area_idx, 1],
                    map_range[area_idx, 2], map_range[area_idx, 3])

# visualize frames in one pattern
frame_vis_flag = False
if frame_vis_flag:
    area_idx = 4
    print('check1')
    frame_in_pattern_vis(map_range[area_idx, 0], map_range[area_idx, 1],
                        map_range[area_idx, 2], map_range[area_idx, 3])

# vis_map_grid = False
# if vis_map_grid:
#     fig = plt.figure(figsize=(30, 30))
#     ax = fig.add_subplot(111)
#     for j in range(map_range.shape[0]):
#         area_idx = j
#         WX, WY, ux_pos, uy_pos = velocity_field_visualization(map_range[area_idx, 0], map_range[area_idx, 1],
#                                                           map_range[area_idx, 2], map_range[area_idx, 3], ax)
#     plt.show()
#     plt.savefig('fig/all.png')
#     pickle.dump(ax, open('myplot.pickle', 'w'))
