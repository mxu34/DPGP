#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from mixtureModel import MixtureModel
import time


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


start_time = time.time()
# dataset = 'NGSIM'
dataset = 'ARGO'

if dataset == 'ARGO':
    # if reinitialize the mixture model
    with open("data_sample/frame_map_range_0_argo_train4", "rb") as np:
        load_frames = pickle.load(np)
    # del load_frames[-1]

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("data_sample/a_mixture_model_ARGO_train4", "wb") as fb:
        pickle.dump(a, fb)

    # # load the saved mixture model
    # with open("data_sample/a_mixture_model_ARGO_train4", "rb") as np:
    #     a = pickle.load(np)

elif dataset == 'NGSIM':

    # if reinitialize the mixture model
    with open("data_sample/frame_US_101_200", "rb") as np: # load saved frames
        load_frames = pickle.load(np)

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("data_sample/a_mixture_model_NGSIM_200", "wb") as fb:
        pickle.dump(a, fb)
    print('direct loaf existing mixture model')

    # load the saved mixture model
    # with open("data_sample/a_mixture_model_NGSIM_200", "rb") as np:  # load saved mixture model
    #     a = pickle.load(np)

for iter in range(3): # Gibbs Sampling Iterations
    # update each frame's indicator/assignment
    a.update_all_assignment()
    # update gaussian process patterns
    a.update_all_pattern()
    # show number of mixtures
    a.show_mixture_model()

if dataset == 'ARGO':
    with open("data_sample/ARGO_final_DPGP_train4_alpha_1", "wb") as fb:
        pickle.dump(a, fb)
else:
    with open("data_sample/NGSIM_200_final_DPGP", "wb") as fb:
        pickle.dump(a, fb)

print('DPGP finished!!')
print("--------------- %s ----------------" % (time.time() - start_time))