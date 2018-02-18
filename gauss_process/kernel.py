#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:44:37 2018

@author: a-scotto
"""

import numpy as np

def get_kernel_function(kernel='sq_exp'):
    
    kernel_func = None
    
    if kernel == 'sq_exp':      
        def kernel_func(d, theta):         
            return np.exp(- np.sum( theta * d**2, axis=(d.ndim - 1) ))
        
    elif kernel == 'ornstein':
        def kernel_func(d, theta):         
            return np.exp(- np.sum( theta * d, axis=(d.ndim - 1) ))
    
    else:
        raise ValueError("Kernel {} not defined. Please refer to existing kernel.".format(kernel))
    
    return kernel, kernel_func