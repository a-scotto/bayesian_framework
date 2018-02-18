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
        param_bounds.append((-3, 5)) 
    
    optimal_theta = differential_evolution(lambda l: cnl_likelihood(y_sample, cov_matrix, l), param_bounds, maxiter=2)
    
    return 10**optimal_theta.x


def cnl_likelihood(y_sample, cov_matrix, length_scale):
    
    length_scale = 10**length_scale
    
    K = cov_matrix.kernel_func(cov_matrix.codist_matrix, length_scale)
    
    try:
        L = lg.cholesky(K)
        
    except lg.LinAlgError:
        return 1e15
    
    modeling_capacity = np.log10(np.abs(np.prod(L.diagonal())))
    data_fit = y_sample.T.dot(lg.solve(L.T, lg.solve(L, y_sample)))
    reg = lg.norm(length_scale)
    
    return 1 / 2 * (modeling_capacity + data_fit) + 1e-3 * reg
    