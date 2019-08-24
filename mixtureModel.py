#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gamma
from frame import Frame
from MotionPattern import MotionPattern
from util import Util
import multiprocessing as mp
from functools import partial
from frame import Frame
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, loggamma


class MixtureModel(object):
    def __init__(self):
        # note the gamma initialization is in util
        # self.gammaShape = 0.0
        # self.gammaScale = 2.0

        self.frames = None
        self.b = None  # Motion Patterns
        self.z = None  # Assignments
        self.K = None  # Number of Motion Patterns
        self.n = None  # number of Data Points
        self.alpha = None  # concentration factor
        self.partition = None  # number of frames in each cluster
        self.parameterIteration = None  # Iteration of Gibbs Sampling
        self.assignmentIteration = None  # Iteration of Gibbs Sampling

    def mixture_model(self, frames):
        self.assignmentIteration = 0
        self.parameterIteration = 0
        # draw patterns one by one
        self.alpha = np.random.gamma(1.0, 1.0)
        self.frames = frames[0]
        self.n = 1
        self.K = 1
        self.z = 1
        # draw a new pattern
        # TODO: pwx pwy not used here
        wx, wy, pwx, pwy = Util.draw_w()
        ux, uy, sigmax, sigmay, sigman = Util.cov_mean(frames[0])
        pattern_0 = MotionPattern(ux, uy, sigmax, sigmay, sigman, wx, wy)
        pattern_0.update_para(frames[0])
        self.b = pattern_0
        # update partition
        self.partition = Util.z2partition(self.z, self.K)
        for i in range(1, len(frames)):
            self.frames.append(frames[i])
            self.n = i
            self.z.append(1)
            self.update_assignment(i)
            print('Initializing Frame: ', i, 'total frames: ', len(frames))

    def update_all_pattern(self):
        # update each motion pattern, wx and wz
        frame_ink_prep = [None] * self.K
        b_prep = [None] * self.K
        for k in range(self.K):
            frame_ink_prep[k] = self.frame_ink(k, 0, True)
            b_prep[k] = self.b[k]
        pool = mp.Pool(mp.cpu_count()) # TODO whether can simplify defining pool
        b_prep = [pool.apply(self.update_one_pattern, args=(frame_ink_prep[k], b_prep[k]))
                  for k in range(self.K)]
        pool.close()
        for k in range(self.K):
            self.b[k] = b_prep[k]
        self.draw_alpha()
        self.parameterIteration += 1

    def draw_alpha(self):
        # note that this part contains a lot of pre-defined parameters.
        pool = mp.Pool(mp.cpu_count())
        # TODO check why there is 20 times here
        candidate = np.linspace(0.1, self.K**2/self.n*20, Util.alpha_sample_size)
        logp = pool.map(self.p_alpha, (c for c in candidate))
        pool.close()
        logp = logp - np.max(logp) # normalization
        if Util.alpha_sample:
            p = np.exp(logp)
            self.alpha = np.random.choice(candidate, 1, p)
        else:
            idx = np.argmax(logp)
            self.alpha = candidate[idx]

    def p_alpha(self, alpha):
        return (self.K - 1.5) * alpha - 0.5 / alpha + np.log(gamma(alpha)/gamma(self.n+alpha))

    def update_one_pattern(self, frame_ink_k, b_prep_k):
        frame_ink = frame_ink_k
        if len(frame_ink.x) > Util.max_obj_update:
            idx= np.random.randint(len(frame_ink.x), Util.max_obj_update)
            frame_ink = Frame(frame_ink.x[idx], frame_ink.y[idx], frame_ink.vx[idx], frame_ink.vy[idx])
        b_prep_k = b_prep_k.update_para(frame_ink)
        return b_prep_k

    def update_all_assignment(self):
        # TODO check parallel process, the same pool will this affect the final result?
        pool = mp.Pool(mp.cpu_count())
        pool.map(self.update_assignment, (i for i in range(self.n)))
        pool.close()
        self.assignmentIteration += 1

    def update_assignment(self, i):
        print('updating assignment: ', i)
        zi_old = self.z[i]
        log_pzik_post = self.assignment_posterior(i)
        # resample zi
        if Util.assignment_MAE:
            self.z[i] = np.argmax(log_pzik_post)
        else:
            log_pzik_post = log_pzik_post - np.max(log_pzik_post)
            pzik_post = np.exp(log_pzik_post)
            pzik_post = pzik_post/np.sum(pzik_post)
            candidate = np.linspace(0, self.K, self.K+1)
            self.z[i] = np.random.choice(candidate, 1, pzik_post)
        # after update zi
        # if zi = K+1, draw a new pattern
        if self.z[i] == self.K+1:
            new_pattern, p_pattern = self.draw_new_pattern()
            # unlike the existing motion pattern, the parameter of
            # the new patterns are updated once generated
            # TODO: check whether this can be appended
            self.b.append(new_pattern.update_para(self.frames[i]))
            self.K += 1
        # if b(zi_old) is empty after the frame_i left
        all_z = set(self.z)
        if not zi_old in all_z:
            # TODO check whether this will happen
            print('delete pattern: ', zi_old)
            # if is empty, delete the pattern
            # TODO check how the pattern will be refilled
            self.b[zi_old] = []
            idx = self.z >= zi_old
            self.z[idx] -= 1
            self.K -= 1
        # update partition
        self.partition = Util.z2partition(self.z, self.K)
        if Util.show_MM:
            self.show_mixture_model()

        # uptate the unchanged pattern parameters
        # TODO check whether can be parallel processed
        for k in range(self.K):
            idx = self.z == k
            ux, uy, sigmax, sigmay, sigman = Util.cov_mean(frames=self.frames[idx])
            self.b[k].ux = ux
            self.b[k].uy = uy
            self.b[k].sigmax = sigmax
            self.b[k].sigmay = sigmay
            self.b[k].sigman = sigman

    def assignment_posterior(self, i):
        # calculate the posterior distribuion of zi
        # calculate the PMF vector
        log_pzik_post = np.zeros((self.K+1, 1))
        # for existing patterns
        # TODO check: parallel processing
        pool = mp.Pool(mp.cpu_count())
        log_post_i = partial(self.log_posterior_exist_pattern, i=i)
        log_pzik_post = pool.map(log_post_i, (k for k in range(self.K)))
        pool.close()
        log_pzik_post.append(self.log_posterior_unseen_pattern(i))
        return log_pzik_post

    def log_posterior_unseen_pattern(self, i):
        # prior
        log_pzik_new_prior = np.log(self.alpha/(self.n - 1 + self.alpha))
        # likelihood : the MCMC integration
        log_pzik_likelihood = self.log_posterior_unseen_pattern(i)
        return log_pzik_new_prior + log_pzik_likelihood

    def log_likelihood_unseen_pattern(self, i):
        # use MC to calculate the integration
        temp_sum = 0
        frame_i = self.frames[i]
        # TODO can be paralleled
        for i in range(Util.mc_iteration):
            new_pattern, p_pattern = self.draw_new_pattern()
            likelihood = new_pattern.GP_prior(frame_i)
            temp_sum += likelihood*p_pattern
        integration = temp_sum/Util.mc_iteration
        return np.log(integration)

    def log_posterior_exist_pattern(self, k, i):
        # calculate the posterior PDF of log_pzik_prior
        partition_without_i = self.partition
        partition_without_i[self.z[i]] -= 1
        log_pzik_prior = np.log(partition_without_i[k]/(self.n - 1 + self.alpha))
        log_pzik_likelihood = self.log_likelihood_exit_pattern(i, k)
        return log_pzik_prior + log_pzik_likelihood

    def log_likelihood_exit_pattern(self, i, k):
        # calculate the log likelihood of frame i under a given pattern k
        # check if frame i is the only frame in bk
        if self.z[i] == k and self.partition[k] == 1:
            # TODO check : self.function may not work
            log_pzik_likelihood = np.log(self.b[k].GP_prior(self.frames[i]))
        else:
            frame_ink = self.frame_ink(k, i)
            # here we approximate the likelihood from the GP field
            # generated by the N_nbr_max nearest observations
            # TODO: note that the ink is sorted by knn by N_nbr_num
            # TODO: note the attention is aapplied by only considering nearest neighbors
            n_nbr = np.min(Util.N_nbr_num, len(frame_ink.x))
            points = np.vstack((frame_ink.x, frame_ink.y))
            knn = NearestNeighbors(n_neighbors=n_nbr, p=1)
            knn.fit(points)
            querry_point = np.array([self.frames[i].x, self.frames[i].y])
            n_idx = knn.kneighbors(querry_point, return_distance=False)
            n_idx = np.unique(n_idx)
            near_frame = Frame(frame_ink.x[n_idx], frame_ink.y[n_idx], frame_ink.vx[n_idx], frame_ink.vy[n_idx])
            frame_i = self.frames[i]
            ux_pos, uy_pos, covx_pos, covy_pos, likelihood = self.b[k].GP_posterior(frame_i, near_frame)
            return np.log(likelihood)

    def frame_ink(self, k, i, all_frame=False):
        # frame ink stores all the data in kth pattern except the ith frame
        Idx = np.where(self.z == k)
        # TODO check whether i exit
        if not all_frame:
            Idx = Idx[Idx != i]
        frames_i = self.frames[Idx]
        # get all concatenation
        frames_i = Frame.combined_frame(frames_i)
        return frames_i

    def draw_new_pattern(self):
        # draw a new motion pattern from the current mixture model
        # pPattern is the pdf of this model
        wx, wy, pwx, pwy = Util.draw_w()
        new_pattern = MotionPattern(self.b[0].ux, self.b[0].uy, self.b[0].sigmax,
                                    self.b[0].sigmay, self.b[0].sigman, wx, wy)
        p_pattern = pwx * pwy
        return new_pattern, p_pattern

    def show_mixture_model(self):
        print('alpha: ', self.alpha, ' n: ', self.n, ' K: ', self.K)
        print('assignmentIter: ', self.assignmentIteration, ' paraIter: ', self.parameterIteration)
        print('----------------------------')


