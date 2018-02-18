#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:59:37 2018

@author: a-scotto
"""
import numpy as np
from sklearn.kernel_ridge import KernelRidge

class Regressor(object):
    
    def __init__(self, x_sample, y_sample, degree=0):
        
        self.reg = KernelRidge(alpha=1e-5, degree=degree)
        self.reg.fit(x_sample, y_sample)


    def fit(self, x_sample, y_sample):
        
        self.reg.fit(x_sample, y_sample)
    
    def predict(self, x):
        
        x = np.atleast_2d(x)
        return self.reg.predict(x)
        