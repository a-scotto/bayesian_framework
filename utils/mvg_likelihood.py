#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:58:29 2018

@author: a-scotto
"""
import numpy as np
import numpy.linalg as lg
from scipy.optimize import differential_evolution

def mvg_likelihood(y_sample, cov_matrix):
    
    n, d = cov_matrix.x_sample.shape
    param_bounds = []
    
    for i in range(d):  
        param_bounds.append((-3, 3)) 
    
    optimal_theta = differential_evolution(lambda l: cnl_likelihood(y_sample, cov_matrix, l), param_bounds, maxiter=3)
    
    return 10**optimal_theta.x


def cnl_likelihood(y_sample, cov_matrix, length_scale, regularize=True, tol=1e15):
    
    length_scale = 10**length_scale
    n = y_sample.size
    
    K = cov_matrix.kernel_func(cov_matrix.codist_matrix, length_scale)
    
    if lg.cond(K) > tol:
        output = lg.cond(K)**0.5
    else :
        try:
            L = lg.cholesky(K)

            modeling_capacity = np.log(np.abs(np.prod(L.diagonal())))
            data_fit = n * np.log(y_sample.T.dot(lg.solve(L.T, lg.solve(L, y_sample))) / n)

            if regularize:
                output = modeling_capacity + data_fit + 1e-2 * lg.norm(length_scale) + 1e-2 * lg.cond(K)**0.25
            else:
                output = modeling_capacity + data_fit

        except lg.LinAlgError:
            output = lg.cond(K)**0.5

    return float(output)
    
    
    