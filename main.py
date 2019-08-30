#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from mixtureModel import MixtureModel

# Main script for DP-GP
# The python version code is implemented by Mengdi Xu, mengdixu@andrew.cmu.edu @SafeAI lab in CMU.
# Initial python code by Yaohui Guo and Vinay Varma Kalidindi
# paper reference:
# Modeling Multi-Vehicle Interaction Scenarios Using Gaussian Random Field
# https://arxiv.org/pdf/1906.10307.pdf
#
# Improvement:
# The python version code fixed several bugs in the MATLAB version of code.
# The code structure is more clear and can easily be implemented for various applications.
#
# Thanks members of SafeAI lab for discussion.
#
# Input:
# frames: list with element as object defined in frame.py
# Output:
# Mixture model as defined in mixtureModel.py


dataset = 'NGSIM'
# dataset = 'ARGO'

if dataset == 'ARGO':
    # if reinitialize the mixture model
    with open("data_sample/map_range_0", "rb") as np:
        load_frames = pickle.load(np)

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("data_sample/a_mixture_model", "wb") as fb:
        pickle.dump(a, fb)

    # load the saved mixture model
    with open("a_mixture_model", "rb") as np:
        a = pickle.load(np)

elif dataset == 'NGSIM':
    # if reinitialize the mixture model
    with open("data_sample/frame_US_101_200", "rb") as np: # load saved frames
        load_frames = pickle.load(np)

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("data_sample/a_mixture_model_NGSIM_200", "wb") as fb:
        pickle.dump(a, fb)

#     # load the saved mixture model
#     with open("a_mixture_model_NGSIM_200", "rb") as np:  # load saved mixture model
#         a = pickle.load(np)

for iter in range(2): # Gibbs Sampling Iterations
    # update each frame's indicator/assignment
    a.update_all_assignment()
    # update gaussian process patterns
    a.update_all_pattern()
    # show number of mixtures
    a.show_mixture_model()

with open("NGSIM_200_final_DPGP", "wb") as fb:
    pickle.dump(a, fb)

print('DPGP finished!!')