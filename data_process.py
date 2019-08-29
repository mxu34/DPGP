#!usr/bin/env python3

import numpy as np
import matplotlib as plt
import os
import pandas as pd
from frame import Frame
import pickle



# Unique object type
# 'VEHICLE' 'LARGE_VEHICLE' 'ON_ROAD_OBSTACLE' 'TRAILER' 'PEDESTRIAN'
#  'BICYCLE' 'BICYCLIST'

# map range with highest traffic density
map_range = np.array([[2570, 2600, 1180, 1210], [2640, 2670, 1240, 1270]])
area_idx = 0

data_path = '/home/mengdi/Dropbox/Research/Mobility21/Mobility21/train4/'

file_names = os.listdir(data_path)
frames = []
for i in range(len(file_names)):
    # open a new file
    data = pd.read_csv(data_path + file_names[i], sep=" ")
    data['log'] = file_names[i]
    # print(data['x'].min(), data['x'].max(), data['y'].min(), data['y'].max())

    # filter by range
    data_range = data[data['x'] >= map_range[area_idx, 0]]
    data_range = data_range[data_range['x'] <= map_range[area_idx, 1]]
    data_range = data_range[data_range['y'] >= map_range[area_idx, 2]]
    data_range = data_range[data_range['y'] <= map_range[area_idx, 3]]

    # load data file into
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
    with open("map_range_0", "wb") as fb:
        pickle.dump(frames, fb)

    with open("map_range_0", "rb") as np:
        load_frames = pickle.load(np)

    print(len(load_frames))
    print(load_frames[0].x.shape)