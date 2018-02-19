#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:06:56 2018

@author: a-scotto
"""
import numpy as np
import numpy.linalg as lg
from utils.ic import maximize_ic
from gauss_process.regressor import Regressor
from gauss_process.cov_matrix import CovarianceMatrix

class GaussianProcess(object):
    
    def __init__(self, x_sample, y_sample, kernel='sq_exp'):
        
        self.x_sample = np.atleast_2d(x_sample)
        self.y_sample = np.atleast_2d(y_sample)
        
        self.cov_matrix = CovarianceMatrix(self.x_sample, kernel)
        self.cov_matrix._fit_hyper_param(self.y_sample)
        self.K = self.cov_matrix.get_covariance_matrix()
        
        self.regressor = Regressor(self.x_sample, self.y_sample)
        
        self.solved = self._solve()
       
    
    def _solve(self):
        
        solved = {}
        n = self.y_sample.size
        e = np.ones((n))
        m_x = self.regressor.predict(self.x_sample)
        
        try:
            L = lg.cholesky(self.K)
            solved['y_sample'] = lg.solve(L.T, lg.solve(L, self.y_sample))
            solved['e'] = lg.solve(L.T, lg.solve(L, e))
            solved['m_x'] = lg.solve(L.T, lg.solve(L, m_x))
            solved['var'] = (self.y_sample - m_x).T.dot(solved['y_sample'] - solved['m_x']) / n
        
        except lg.LinAlgError:
            solved['y_sample'] = lg.solve(self.K, self.y_sample)
            solved['e'] = lg.solve(self.K, e)
            solved['m_x'] = lg.solve(self.K, m_x)
            solved['var'] = (self.y_sample - m_x).T.dot(solved['y_sample'] - solved['m_x']) / n
            
        return solved
    
    
    def infer_value(self, x):
        
        n, d = self.x_sample.shape
        x = np.atleast_2d(x)
        
        if x.size != d:
            raise ValueError('x vector (dim={}) must have same dimension as sampled points (dim={}).'.format(x.size, d))
        
        d = np.abs(x - self.x_sample)        
        r = self.cov_matrix.kernel_func(d, self.cov_matrix.theta)
        
        return self.regressor.predict(x) + r.dot(self.solved['y_sample'] - self.solved['m_x'])
        
   
    def estimate_mse(self, x):
        
        n, d = self.x_sample.shape
        x = np.atleast_2d(x)
        
        if x.size != d:
            raise ValueError('x vector (dim={}) must have same dimension as sampled points (dim={}).'.format(x.size, d))
        
        d = np.abs(x - self.x_sample)
        r = self.cov_matrix.kernel_func(d, self.cov_matrix.theta)
        r_solved = lg.solve(self.K, r)
        
        e = np.ones((self.y_sample.size))
        
        a = float(e.T.dot(self.solved['e']))
        b = float(r.T.dot(r_solved))
        c = float(e.T.dot(r_solved))
        
        s = self.solved['var'] * (1 - b + (1 - c)**2 / a)
        
        return max(0, s)
    
    def enrich_model(self, bounds, crit='exp_imp'):
        
        x_next = maximize_ic(self, bounds, crit)
        
        return x_next
    
    def _fit_process(self):
        
        self.cov_matrix._fit_hyper_param(self.y_sample)
        self.K = self.cov_matrix.get_covariance_matrix()
        self.regressor.fit(self.x_sample, self.y_sample)
        self.solved = self._solve()
    
    def add_sample(self, x_new, y_new):
        
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        
        self.x_sample = np.vstack([self.x_sample, x_new])
        self.y_sample = np.vstack([self.y_sample, y_new])
        
        self.cov_matrix.add_sample(x_new)
        self._fit_process()
    
    def remove_sample(self, index):
        
        self.y_sample = np.delete(self.y_sample, index, axis=0)
        self.x_sample = np.delete(self.x_sample, index, axis=0)
        
        self.cov_matrix.remove_sample(index)
        self._fit_process()