# DPGP for Multi-Vehicle Interaction Scenario Extraction
The clustering results on NGSIM and Argoverse are coming soon. <br>
This repo provides the python implementation of DPGP algorithm using Gaussian Process to represent multi-vehicle driving scenarios with Dirichlet Process adapting cluster numbers. <br>
The python version code is implemented by Mengdi Xu, mengdixu@andrew.cmu.edu @SafeAI lab in CMU. <br>
Initial MATLAB code implemented by Yaohui Guo and Vinay Varma Kalidindi. <br>

### Paper Reference:
Modeling Multi-Vehicle Interaction Scenarios Using Gaussian Random Field <br>
https://arxiv.org/pdf/1906.10307.pdf


### Improvement:
(a) fixed several bugs in the MATLAB version of code. <br>
(b) The code structure is more clear and can easily be implemented for various applications. <br>
Thanks members of SafeAI lab for discussion! <br>


#### Input:

frames: list with element as object defined in frame.py <br>

#### Output:

Mixture model as defined in mixtureModel.py <br>

#### Implement:
Train DPGP: python main.py <br>
Visualization: python pattern_vis.py

#### Required python packages:
numpy         == 1.16.4 <br>
scipy         == 1.3.1 <br>
scikit-learn  == 0.21.2 <br>
pandas        == 0.25.0 <br>
<math 
multiprocessing
functools
pickle>
