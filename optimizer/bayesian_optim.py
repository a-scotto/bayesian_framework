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


    def initialize(self, x_sample, obj_func):
        
        obj_func.reset()
        self.obj_func = obj_func
        
        y_sample = np.zeros((x_sample.shape[0], 1))    
        for i, xi in enumerate(x_sample):
            y_sample[i] = obj_func.evaluate(xi)
            
        self.gp = GaussianProcess(x_sample, y_sample, self.param['kernel'])
    
    def _set_bounds(self, x_local, step_size):
        
        bounds = []
        
        for i, x_local_i in enumerate(x_local):
            inf = 0
            sup = 0
            
            if x_local_i - step_size < self.obj_func.bounds[i][0]:
                inf = self.obj_func.bounds[i][0]
            else:
                inf = x_local_i - step_size
            
            if x_local_i + step_size > self.obj_func.bounds[i][1]:
                sup = self.obj_func.bounds[i][1]
            else:
                sup = x_local_i + step_size
            
            bounds.append((inf, sup))
        
        return bounds

    def run(self, max_budget=500, tolerance=1e-5):

        k = 0

        step_size = self.param['init_step_size']     
        updater = Updater(self.gp, self.param)
        global_bounds = self.obj_func.bounds

        while step_size > tolerance and self.obj_func.eval_count < max_budget:
            
            local_bounds = self._set_bounds(updater.x_local, step_size)
            
            if self.param['mode'] in ['ego', 'elgo']:
                x_trial = self.gp.enrich_model(global_bounds)
            
            else:
                x_trial = self.gp.enrich_model(local_bounds)
            
            y_trial = self.obj_func.evaluate(x_trial)
            
            step_size = updater.update_step_size(x_trial, y_trial, self.obj_func, local_bounds)
            
            self.gp.add_sample(x_trial, y_trial)
            
            if self.gp.y_sample.size > 10 * (self.obj_func.func_id[1] + 1):
                self.gp.remove_sample(self.gp.y_sample.argmax())
            
            k = k + 1
            
            if self.param['verbosity']:
                print("########       Iteration {}       ########".format(k))
                print("#")
                print("#     Function Evaluations = ", self.obj_func.eval_count)
                print("#     Step size = ", step_size)
                print("#     Model size = ", self.gp.y_sample.size)
                print("#     Found value = ", y_trial)
                print("#     Point found = ", x_trial)
                print("#     Local point = ", updater.x_local)
                print("#     Best value = ", min(self.obj_func.evals))
                print()

            
#            # Model size control
#            if model.Y.size > min(12 * (self.f.varNumber + 1), 120):
#                model.removePoint(model.Y.argmax())
#            if self.mode == "ELGOWL":
#                if localModel.Y.size > 2 * (self.f.varNumber + 1):
#                    localModel.removePoint(localModel.Y.argmax())
#
#            if verbosity:
#                print("########       Iteration {} : {}s       ########".format(k, round(T[-1], 2)))
#                print("#")
#                print("#     Function Evaluations = ", self.f.evalCounter)
#                print("#     Model size = ", model.Y.size)
#
#                if self.mode in ["EGO"]:
#                    print("#     Iteration Value = ", yTrial)
#                    print("###   Current Minimum Value = ", min(self.f.evaluations), "   ##\n")
#
#                elif self.mode in ["GCEGO", "GCEGOrestart", "TRIKE", "TRIKErestart"]:
#                    print("#     Radius = ", searchLimit)
#                    print("#     Iteration Value = ", yTrial)
#                    print("#     Local Center Value = ", yLocalCenter)
#                    print("###   Current Minimum Value = ", min(self.f.evaluations), "   ##\n")
#
#                elif self.mode in ["ELGO", "ELGOWL"]:
#                    print("#     Local model size = ", localModel.Y.size)
#                    print("#     Radius = ", searchLimit)
#                    print("#     Main Loop Value = ", yTrial)
#                    print("#     Second Loop Value = ", localModel.Y[-1, 0])
#                    print("#     local Center Value = ", yLocalCenter)
#                    print("###   Current Minimum Value = ", min(self.f.evaluations), "   ##\n")

        evals = np.asarray(self.obj_func.evals)
        cost = self.obj_func.eval_count
        x_opt = self.gp.x_sample[self.gp.y_sample.argmin()]

        return evals, cost, x_opt