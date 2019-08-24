#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy
from util import Util
from scipy.stats import gamma
from scipy.stats import multivariate_normal
import numpy.matlib
from numpy.linalg import inv


class MotionPattern(object):
    def __init__(self, ux=0.0, uy=0.0, sigmax=1.0, sigmay=1.0, sigman=1.0, wx=1.0, wy=1.0):
        self.ux = ux
        self.uy = uy
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.sigman = sigman
        self.wx = wx
        self.wy = wy

    def update_para(self, frames):
        # search the best wx,wy parameters for its assigned frames and
        # return the motion pattern with the updated parameters

        if not Util.useMLE:
            try:
                self.update_para_sample(frames)
            except:
                wx, wy, pwx, pwy = Util.draw_w()
                self.wx = wx
                self.wy = wy
        else:
            self.update_para_MLE(frames)

    def update_para_sample(self, frames):
        x = np.linspace(1,51,1)
        WX = np.meshgrid(x)
        WX = np.reshape(WX, (-1, 1))
        WY = WX
        # prior
        PWX = gamma.pdf(WX, a=Util.gammaShape, scale=Util.gammaScale)
        PWY = gamma.pdf(WY, a=Util.gammaShape, scale=Util.gammaScale)
        log_PWXWY_prior = np.log(np.multiply(PWX, PWY))
        # likelihood
        log_PWXWY_likelihood = np.zeros(len(log_PWXWY_prior))
        for i in range(len(WX)):
            self.wx = WX[i]
            self.wy = WY[i]
            log_PWXWY_likelihood[i] = np.log(self.GP_prior(frames))
        # posterior
        log_PWXWY_post = log_PWXWY_prior + log_PWXWY_likelihood
        log_PWXWY_post = log_PWXWY_post - max(log_PWXWY_post) # why - max
        PWXWY_post = np.exp(log_PWXWY_post)
        # resample based on posterior prob
        candidate = np.linspace(0, len(PWXWY_post)-1, len(PWXWY_post))
        idx = np.random.choice(candidate, 1, PWXWY_post)
        # uptate wx wy
        self.wy = WY[int(idx)]
        self.wx = WX[int(idx)]

    def update_para_MLE(self, frames):
        # not used in algorithm
        pass

    def squared_exp_cov(self, x1, y1, x2, y2, bnoise):
        # calculate the covariance matrix given location data and motion
        # pattern. x,y are column vectors
        X2, X1 = np.meshgrid(x2, x1)
        Y2, Y1 = np.meshgrid(y2, y1)

        # Change to meshgrid instead of using repmat
        # m = len(x1)
        # n = len(x2)
        # X1 = np.matlib.repmat(x1,1,n)
        # Y1 = np.matlib.repmat(y1,1,n)
        # X2 = np.matlib.repmat(np.transpose(x2), m, 1)mengdixu
        # Y2 = np.matlib.repmat(np.transpose(y2), m, 1)

        disMat = -(X1-X2)**2/(2*self.wx**2) - (Y1-Y2)**2/(2*self.wy**2)
        if bnoise:
            xK = self.sigmax**2 * np.exp(disMat) + self.sigman**2 * np.eye(len(x1))
            yK = self.sigmay**2 * np.exp(disMat) + self.sigman**2 * np.eye(len(x1))
        else:
            xK = self.sigmax**2 * np.exp(disMat)
            yK = self.sigmay**2 * np.exp(disMat)
        return xK, yK

    def GP_posterior(self, frame_test, frame_train):
        # calculate the likelihood of a frame under motion patter with
        # given data.
        # TODO check GP_posterior is calculating likelihood???
        # x,y: frame testing (with *)
        # X,Y: frame training(no *)
        # TODO check why bnoise is true and false
        xKXYXY, yKXYXY = self.squared_exp_cov(frame_train.x, frame_train.y, frame_train.x, frame_train.y, True)
        xKxyxy, yKxyxy = self.squared_exp_cov(frame_test.x, frame_test.y, frame_test.x, frame_test.y, False)
        xKxyXY, yKxyXY = self.squared_exp_cov(frame_test.x, frame_test.y, frame_train.x, frame_train.y, False)
        xKXYxy = np.transpose(xKxyXY)
        yKXYxy = np.transpose(yKxyXY)

        xtemp = np.dot(xKxyXY, inv(xKXYXY))
        ytemp = np.dot(yKxyXY, inv(yKXYXY))
        ux_pos = self.ux * np.ones_like(frame_test.x) + xtemp*(frame_train.vx - self.ux*np.ones_like(frame_train.x))
        uy_pos = self.uy * np.ones_like(frame_test.y) + ytemp*(frame_train.vy - self.uy*np.ones_like(frame_train.y))

        covx_pos = xKxyxy - np.dot(xtemp, xKXYxy)
        covy_pos = yKxyxy - np.dot(ytemp, yKXYxy)

        covx_pos = (covx_pos + np.transpose(covx_pos)) / 2.0 + \
                   Util.eip_post * np.eye(covx_pos.shape[0], covx_pos.shape[1])
        covy_pos = (covy_pos + np.transpose(covy_pos)) / 2.0 + \
                   Util.eip_post * np.eye(covy_pos.shape[0], covy_pos.shape[1])
        likelihood = multivariate_normal(frame_test.vx, ux_pos, covx_pos) * \
                       multivariate_normal(frame_test.vy, uy_pos, covy_pos)
        return ux_pos, uy_pos, covx_pos, covy_pos, likelihood

    def GP_prior(self, framesTest):
        # calculate the likelihood of a testing frame under a GP
        # without observing any data
        # TODO check whether bnoise should always be False
        covx, covy = self.squared_exp_cov(framesTest.x, framesTest.y, framesTest.x, framesTest.y, False)
        covx = (covx + np.transpose(covx)) / 2.0 + Util.eip_prior * np.eye(covx.shape[0], covx.shape[1])
        covy = (covy + np.transpose(covy)) / 2.0 + Util.eip_prior * np.eye(covy.shape[0], covy.shape[1])
        ux_prior = self.ux * np.ones_like(framesTest.vx)
        uy_prior = self.uy * np.ones_like(framesTest.vy)
        likelihood = multivariate_normal(framesTest.vx, ux_prior, covx) * multivariate_normal(framesTest.vy, uy_prior, covy)
        return likelihood


