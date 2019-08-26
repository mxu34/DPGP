#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gamma
from frame import Frame
from MotionPattern import MotionPattern
from mixtureModel import MixtureModel
from functools import partial
import multiprocessing as mp


class test(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update(self, a, b):
        self.x = a
        self.y = b

def job(x, y):
    return x*y


pool = mp.Pool(mp.cpu_count())
y = 1
log_post_i = partial(job, y=y)
k = np.linspace(1, 10, 10)
log_pzik_post = pool.map(log_post_i, (k for k in range(10)))
print(log_pzik_post)

obj = test(1,2)
print(np.arange(1,50,1))
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])
vx = np.array([1,2,3,4])
vy = np.array([1,2,3,4])

frame_1 = Frame(x,y,vx,vy)
frame_2 = Frame(x,y,vx,vy)
frames = [frame_1, frame_2]

print(frame_2.x + frame_1.x)
frame_new_x = np.concatenate((frame_2.x, frame_1.x), axis = 0)
x = y= []
x = np.concatenate((x, frame_new_x), axis = 0)
print(np.vstack((frame_2.x, frame_2.y)))
#
# ux= uy=sigmax=sigmay=sigman= wx= wy= 0.0
# b = []
# c = MotionPattern(ux, uy, sigmax, sigmay, sigman, wx, wy)
# b.append(c)
# b.append(c)
# print('b shape',len(b))
# b[0].ux = 2
# print('b first element', b[0].ux)
# d = []
# d = 10
# print('d', d)
# print(x)
# print(y)
# print(len(frames))
# print(frames[0].x.shape)
#
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = np.exp(a)
# c = np.eye(b.shape[0], b.shape[1])
# print('c', c)