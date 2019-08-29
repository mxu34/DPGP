#!usr/bin/env python3

import os
import numpy as np
from pykalman import KalmanFilter
import pandas as np
import matplotlib as plt

datadir = '/home/mengdi/Dropbox/Research/Mobility21/Mobility21/train3/'

all_files = os.listdir(datadir)
# for i in range(len(all_files)):
for i in range(0):
    name = datadir + all_files[0]
    content = pd.read_csv(name, sep=" ")
    label_list = content.label_class.unique()
    #     for i in range(len(label_list)):
    for i in range(len(label_list)):
        obj_df = content[content['label_class'] == label_list[i]]

        x = content.x.values
        to_idx = np.max(len(x), 3)
        mean_init = np.mean(x[0:to_idx])
        kf = KalmanFilter(n_dim_obs=1,
                          n_dim_state=1,
                          initial_state_mean=mean_init,
                          initial_state_covariance=2,
                          transition_matrices=[1],
                          transition_covariance=np.eye(1),
                          transition_offsets=None,
                          observation_matrices=[1],
                          observation_covariance=2
                          )
        x_mean, cov = kf.filter(x)
        plt.scatter(x_mean)
        plt.show()
    content.to_csv(name, sep=' ', index=False, header=True)