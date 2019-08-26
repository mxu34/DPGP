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
