#!usr/bin/env python3

import numpy as np
import matplotlib as plt
import os
import pandas as pd
from frame import Frame
import pickle


# Preprocess NGSIM dataset into frames
# Structure of NGSIM source data
# data_path
# -- log_name.txt
# -- -- [frame_idx x y vel_x vel_y]

data_path = '/home/mengdi/Dropbox/Research/Mobility21/DPGP/DPGP_Mobility21/DPGP_Uber/frameUS-101.txt'

data_range = pd.read_csv(data_path, sep="\t", index_col=False)

frames = []
if data_range.shape[0] != 0:
    frame_range_idx = data_range['frame_idx'].unique()
    for j in range(len(frame_range_idx)):
        # get data at frame_idx j and append to frames
        data_temp = data_range[data_range['frame_idx'] == frame_range_idx[j]]
        frame_temp = Frame(data_temp.x.values, data_temp.y.values,
                           data_temp.vel_x.values, data_temp.vel_y.values)
        frames.append(frame_temp)
        del frame_temp

print(len(frames))
print(frames[0].x.shape)
with open("frame_US_101_200", "wb") as fb:
    pickle.dump(frames, fb)
