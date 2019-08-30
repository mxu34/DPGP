#!usr/bin/env python3

import numpy as np
from scipy.stats import gamma


# Util is a class with predefined parameters in DP-GP

class Util(object):
    def __init__(self):
        self.gammaShape = 10.0 # gamma shape parameter for draw w
        self.gammaScale = 2.0 # gamma scale parameter for draw w
        self.eip_prior = 0.000000001 # noise scale parameter for prior
        self.eip_post = 0.0001 # noise scale parameter for posterior
        self.N_nbr_num = 5 # number of nearest neighbors in knn
        self.mc_iteration = 100 # MC iteration for unseen pattern likelihood calculation
        self.useMLE = False # whether use Maximum likelihood for update wx and wy
        self.assignment_MAP = True # whether use Maximum a posterior to update z(assignment)
        self.alpha_MAP = True # whether use Maximum a posterior to update alpha
        self.show_MM = True # show info of mixture models
        self.max_obj_update = 200 # max number of data points used to update each motion pattern
        self.alpha_max = 20 # max boundary of alpha
        self.alpha_min = 0.001 # min boundary of alpha
        self.alpha_sample_size = 200 # sample number between max and min boundary of alpha
        self.alpha_initial_sample = True # whether to use random sample from inverse gamma distribution
        self.alpha_initial = 1 # predefine initial alpha if not sample
        self.sigman = 0.2 # noise covariance scalar in kernel function

    def draw_w(self):
        # randomly draw wx and wy from gamma distribution
        wx = np.random.gamma(self.gammaShape, self.gammaScale)
        wy = np.random.gamma(self.gammaShape, self.gammaScale)
        pwx = gamma.pdf(wx, a=self.gammaShape, scale=self.gammaScale) # prob of wx
        pwy = gamma.pdf(wy, a=self.gammaShape, scale=self.gammaScale) # prob of wy
        return wx, wy, pwx, pwy

