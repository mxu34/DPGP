#!usr/bin/env python3

import numpy as np
import scipy.linalg


def squared_exp_cov_test( x1, y1, x2, y2, bnoise):
    # calculate the covariance matrix given location data and motion
    # pattern. x,y are column vectors
    wx = 1
    wy = 1
    sigmax = 1
    sigmay = 1
    sigman = 1

    X2, X1 = np.meshgrid(x2, x1)
    Y2, Y1 = np.meshgrid(y2, y1)

    disMat = -(X1 - X2) ** 2 / (2 * wx ** 2) - (Y1 - Y2) ** 2 / (2 * wy ** 2)
    if bnoise:
        xK = sigmax ** 2 * np.exp(disMat) + sigman ** 2 * np.eye(len(x1))
        yK = sigmay ** 2 * np.exp(disMat) + sigman ** 2 * np.eye(len(x1))
    else:
        xK = sigmax ** 2 * np.exp(disMat)
        yK = sigmay ** 2 * np.exp(disMat)

    xK = xK[0:len(x1), 0:len(y1)]
    yK = yK[0:len(x1), 0:len(y1)]
    return xK, yK

x1 = np.array([1,2,3,4,5])
y1 = np.array([1,2,3,4,5])

xK, yK = squared_exp_cov_test( x1, y1, x1, y1, False)
s1, v = scipy.linalg.eigh(xK)
s2, v = scipy.linalg.eigh(yK)
print('haha')

