#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from mixtureModel import MixtureModel

# main script for DP-GP

dataset = 'NGSIM'
# dataset = 'ARGO'

if dataset == 'ARGO':
    # if reinitialize the mixture model
    with open("map_range_0", "rb") as np:
        load_frames = pickle.load(np)

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("a_mixture_model", "wb") as fb:
        pickle.dump(a, fb)

    # load the saved mixture model
    with open("a_mixture_model", "rb") as np:
        a = pickle.load(np)

elif dataset == 'NGSIM':
    # if reinitialize the mixture model
    with open("frame_US_101_200", "rb") as np: # load saved frames
        load_frames = pickle.load(np)

    a = MixtureModel(load_frames)
    a.mixture_model()

    # save the initialized mixture model
    with open("a_mixture_model_NGSIM_200", "wb") as fb:
        pickle.dump(a, fb)

    # load the saved mixture model
    with open("a_mixture_model_NGSIM_200", "rb") as np:  # load saved mixture model
        a = pickle.load(np)

for iter in range(10):
    # update each frame's indicator/assignment
    a.update_all_assignment()
    # update gaussian process patterns
    a.update_all_pattern()
    # show number of mixtures
    a.show_mixture_model()

with open("NGSIM_200_final_DPGP", "wb") as fb:
    pickle.dump(a, fb)

print('yeah!!')