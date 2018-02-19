#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:31:51 2018

@author: a-scotto
"""

import numpy as np
from optimizer.updater import Updater
from gauss_process.gauss_process import GaussianProcess

class BayesianOptimizer(object):
    
    def __init__(self, param):
        
        self.param = param.param


    def _initialize(self, x_sample, obj_func):
        
        obj_func.reset()
        
        y_sample = np.zeros((x_sample.shape[0], 1))    
        for i, xi in enumerate(x_sample):
            y_sample[i] = obj_func.evaluate(xi)
            
        return GaussianProcess(x_sample, y_sample, self.param['kernel'])
    
    
    def _set_bounds(self, x_local, step_size, global_bounds):
        
        bounds = []
        
        for i, x_local_i in enumerate(x_local):
            inf = 0
            sup = 0
            
            if x_local_i - step_size < global_bounds[i][0]:
                inf = global_bounds[i][0]
            else:
                inf = x_local_i - step_size
            
            if x_local_i + step_size > global_bounds[i][1]:
                sup = global_bounds[i][1]
            else:
                sup = x_local_i + step_size
            
            bounds.append((inf, sup))
        
        return bounds


    def run(self, obj_func, x_sample, max_budget=500, tolerance=1e-5):

        k = 0
        
        gp = self._initialize(x_sample, obj_func)
        step_size = self.param['init_step_size']     
        updater = Updater(gp, self.param)
        global_bounds = obj_func.bounds

        while step_size > tolerance and obj_func.eval_count < max_budget:
            
            local_bounds = self._set_bounds(updater.x_local, step_size, global_bounds)
            
            if self.param['mode'] in ['ego', 'elgo', 'elgowlm']:
                x_trial = gp.enrich_model(global_bounds)
            
            else:
                x_trial = gp.enrich_model(local_bounds)
            
            y_trial = obj_func.evaluate(x_trial)
            
            step_size = updater.update_step_size(x_trial, y_trial, obj_func, local_bounds)
            
            gp.add_sample(x_trial, y_trial)
            
            if gp.y_sample.size > 10 * (obj_func.func_id[1] + 1):
                gp.remove_sample(gp.y_sample.argmax())
                if self.param['mode'] == 'elgo':
                    gp.remove_sample(gp.y_sample.argmax())
            
            k = k + 1
            
            if self.param['verbosity']:
                print("########       Iteration {}       ########".format(k))
                print("#")
                print("#     Function Evaluations = ", obj_func.eval_count)
                print("#     Step size = ", step_size)
                print("#       Global model size = ", gp.y_sample.size)
                print("#       Global value found = ", y_trial)
                if self.param['mode'] == 'elgowlm':
                    print("#       Local model size = ", updater.gp_local.y_sample.size)
                    print("#       Local value found = ", updater.gp_local.y_sample[-1])
                print("#     Current center = ", updater.y_local)
                print("#     Best value = ", min(obj_func.evals))
                print()

        evals = np.asarray(obj_func.evals)
        cost = obj_func.eval_count
        x_opt = gp.x_sample[gp.y_sample.argmin()]

        return evals, cost, x_opt