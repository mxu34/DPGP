#!usr/bin/env python3

import matplotlib.pyplot as plt
import pickle


# with open("map_range_0", "rb") as np:
#     load_frames = pickle.load(np)

# a = MixtureModel(load_frames)
# a.mixture_model()

# with open("a_mixture_model", "wb") as fb:
#     pickle.dump(a, fb)

with open("a_mixture_model", "rb") as np:
    a = pickle.load(np)

for iter in range(200):
    a.update_all_assignment()
    a.update_all_pattern()
    a.show_mixture_model()
print('haha')