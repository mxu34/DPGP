#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gamma


class Frame(object):
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    @staticmethod
    def combined_frame(self, frames):
        x_ink = y_ink = xv_ink = vy_ink = []
        for i in range(len(frames)):
            x_ink = np.concatenate((x_ink, frames[i].x), axis=0)
            y_ink = np.concatenate((y_ink, frames[i].y), axis=0)
            vx_ink = np.concatenate((vx_ink, frames[i].vx), axis=0)
            vy_ink = np.concatenate((vy_ink, frames[i].vy), axis=0)
        return Frame(x_ink, y_ink, vx_ink, vy_ink)
