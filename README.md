# DPGP for Multi-Vehicle Interaction Scenario Extraction
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


#### Required python packages:
math, numpy, scipy, multiprocessing, functools, sklearn, pandas, pickle
