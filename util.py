#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gamma
from frame import Frame


class Util(object):
    def __init__(self):
        self.gammaShape = 10.0
        self.gammaScale = 2.0
        self.eip_prior = 0.000000001
        self.eip_post = 0.0001
        self.N_nbr_num = 5
        self.mc_iteration = 100
        self.useMLE = False
        self.assignment_MAE = True
        self.alpha_MAE = False # sample
        self.show_MM = True
        self.max_obj_update = 200
        self.alpha_sample_size = 300
        self.alpha_sample = True
        self.sigman = 0.2
        self.alpha_max = 20

    def draw_w(self):
        wx = np.random.gamma(self.gammaShape, self.gammaScale)
        wy = np.random.gamma(self.gammaShape, self.gammaScale)
        pwx = gamma.pdf(wx, a=self.gammaShape, scale=self.gammaScale)
        pwy = gamma.pdf(wy, a=self.gammaShape, scale=self.gammaScale)
        return wx, wy, pwx, pwy

    def cov_mean(self, frames):
        #TODO: sigman doesn't seem to be used
        combinedframes = self.combined_frame(frames)
        ux = np.mean(combinedframes.vx)
        uy = np.mean(combinedframes.vy)
        sigmax = np.std(combinedframes.vx)
        sigmay = np.std(combinedframes.vy)
        return ux, uy, sigmax, sigmay, self.sigman

    @staticmethod
    def z2partition(z, k):
        # convert assignment z to partition
        # partition = np.zeros((k, 1))
        if len(z) >= 120:
            pass
        partition = [0] * k
        for i in range(len(z)):
            partition[z[i]] += 1
        return partition

    @staticmethod
    def combined_frame(frames):
        x_ink = y_ink = vx_ink = vy_ink = []
        if len(frames) == 1:
            return frames[0]
        else:
            for i in range(len(frames)):
                x_ink = np.concatenate((x_ink, frames[i].x), axis=0)
                y_ink = np.concatenate((y_ink, frames[i].y), axis=0)
                vx_ink = np.concatenate((vx_ink, frames[i].vx), axis=0)
                vy_ink = np.concatenate((vy_ink, frames[i].vy), axis=0)
            return Frame(x_ink, y_ink, vx_ink, vy_ink)
