#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:26:05 2018

@author: a-scotto
"""

import numpy as np
import numpy.linalg as lg
from gauss_process.kernel import get_kernel_function
from utils.mvg_likelihood import mvg_likelihood

class CovarianceMatrix(object):
    
    def __init__(self, x_sample, kernel='sq_exp'):
        
        self.x_sample = x_sample
        self.codist_matrix = self._build_codist_matrix()
        self.kernel, self.kernel_func = get_kernel_function(kernel)
        self.theta = None
    
    def _build_codist_matrix(self):
    
        n, d = self.x_sample.shape
        codist_matrix = np.zeros((n, n, d))
        
        for i, xi in enumerate(self.x_sample):   
            for j, xj in enumerate(self.x_sample[:i]):
                codist_matrix[i, j, :] = codist_matrix[j, i, :] = np.abs(xi - xj)
        
        return codist_matrix
    
    def _fit_hyper_param(self, y_sample):
        
        if self.kernel not in ['sq_exp', 'ornstein']:
            print('Kernel chosen do not require any fitting to sampled data.')
        
        else:
            self.theta = mvg_likelihood(y_sample, self)
    
    def get_covariance_matrix(self):
        
        if self.theta is None:
            raise ValueError('Hyper-parameters need to be fitted before computing covariance matrix. Use _fit_hyper_param function.')
        
        K = self.kernel_func(self.codist_matrix, self.theta)
        
        if lg.cond(K) > 10e6:
            K = K + 1e-6 * np.eye(K.shape[0])
        
        return K
    
    def add_sample(self, x_new):
        
        n, d = self.x_sample.shape
        
        d_add = np.abs(x_new - self.x_sample).reshape((n, 1, d))
        self.codist_matrix = np.hstack([self.codist_matrix, d_add])      
        d_add = d_add.reshape((1, n, d))
        
        d_1 = np.hstack([d_add, np.zeros((1, 1, d))])
        self.codist_matrix = np.vstack([self.codist_matrix, d_1])
  
        self.x_sample = np.vstack([self.x_sample, x_new])
    
    def remove_sample(self, index):
        
        self.codist_matrix = np.delete(self.codist_matrix, index, axis=0)
        self.codist_matrix = np.delete(self.codist_matrix, index, axis=1)
        
        self.x_sample = np.delete(self.x_sample, index, axis=0)
        