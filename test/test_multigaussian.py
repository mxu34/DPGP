#!usr/bin/env python3

import numpy as np
import scipy.linalg
import math

def norm_pdf_multivariate( x, mu, sigma):
    # TODO need to check whether this is correct
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        # print('det', det)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0 / (math.pow((2 * np.pi), size / 2.0) * math.pow(det, 0.5))
        x_mu = np.array(x - mu)
        inv = np.linalg.inv(sigma)
        inner = np.dot(x_mu, inv)
        outer = np.dot(inner, np.transpose(x_mu))
        result = math.pow(math.e, -0.5 * outer)
        final = norm_const * result
        return final
    else:
        raise NameError("The dimensions of the input don't match")

sigma = np.array([[4,6,0,8,10],
                  [6,26,17,16,22],
                  [0,17,38,30,20],
                  [8,16,30,57,37],
                  [10,22,20,37,37]])
x = np.array([1,2,3,4,5])
mu = np.array([0,0,2,1,3])
print(norm_pdf_multivariate( x, mu, sigma))
# checked! the same with matlab mvnpdf