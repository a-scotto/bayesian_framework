#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:48:49 2018

@author: a.scotto
"""

import numpy as np
from utils.ic import _compute_ic
from utils.sampling import lhs_sampling
from gauss_process.gauss_process import GaussianProcess

class Updater(object):

    def __init__(self, gaussian_process, parameters):

        self.gp = gaussian_process
        self.param = parameters
        
        if self.param['mode'] in ['elgowlm', 'elgowlmrestart']:
            self.init_x_sample = self.gp.x_sample
            self.init_y_sample = self.gp.y_sample
            self.gp_local = GaussianProcess(self.init_x_sample, self.init_y_sample)
        
        self.x_local = self.gp.x_sample[self.gp.y_sample.argmin()]
        self.y_local = self.gp.y_sample.min()
        
        self.step_size = self.param['init_step_size']
        

    def enforcing_func(self, x):

        return 10**(-2) * (x / self.param['init_step_size'])**(2)


    def _set_step_size(self, succesful):

        if succesful:
            self.step_size *= self.param['incr_fact']
        else:
            self.step_size *= self.param['decr_fact']


    def update_step_size(self, x_trial, y_trial, obj_func, local_bounds):

        if self.param['mode'] == 'ego':
            pass


        if self.param['mode'] == 'gcego':

            if y_trial <= self.y_local - self.enforcing_func(self.step_size):
                self.x_local = x_trial
                self.y_local = y_trial
                self._set_step_size(True)

            else:
                self._set_step_size(False)
        
        
        if self.param['mode'] == 'elgo':

            if y_trial <= self.y_local - self.enforcing_func(self.step_size):
                self.x_local = x_trial
                self.y_local = y_trial
                self.step_size = self.param['init_step_size']

            else:
                x = self.gp.enrich_model(local_bounds)
                y = obj_func.evaluate(x)
                
                if y <= self.y_local - self.enforcing_func(self.step_size):
                    self.x_local = x
                    self.y_local = y
                    self._set_step_size(True)
                else:
                    self._set_step_size(False)

                self.gp.add_sample(x, y)
        
        
        if self.param['mode'] == 'elgowlm':

            if y_trial <= self.y_local - self.enforcing_func(self.step_size):
                self.x_local = x_trial
                self.y_local = y_trial
                self.step_size = self.param['init_step_size']
                self.gp_local.add_sample(x_trial, y_trial)

            else:
                x = self.gp_local.enrich_model(local_bounds)
                y = obj_func.evaluate(x)
                
                if y <= self.y_local - self.enforcing_func(self.step_size):
                    self.x_local = x
                    self.y_local = y
                    self._set_step_size(True)
                else:
                    self._set_step_size(False)
                
                self.gp_local.add_sample(x, y)
                
                if self.gp_local.y_sample.size > 8 * (obj_func.func_id[1] + 1):
                    self.gp_local.remove_sample(self.gp_local.y_sample.argmax())


        if self.param['mode'] == 'elgowlmrestart':

            if y_trial <= self.y_local - self.enforcing_func(self.step_size):
                self.x_local = x_trial
                self.y_local = y_trial
                self.step_size = self.param['init_step_size']
                self.gp_local.__init__(self.init_x_sample, self.init_y_sample)
                self.gp_local.add_sample(x_trial, y_trial)

            else:
                x = self.gp_local.enrich_model(local_bounds)
                y = obj_func.evaluate(x)
                
                if y <= self.y_local - self.enforcing_func(self.step_size):
                    self.x_local = x
                    self.y_local = y
                    self._set_step_size(True)
                else:
                    self._set_step_size(False)
                
                self.gp_local.add_sample(x, y)
                
                if self.gp_local.y_sample.size > 8 * (obj_func.func_id[1] + 1):
                    self.gp_local.remove_sample(self.gp_local.y_sample.argmax())


        if self.param['mode'] == 'trike':

            actual_imp = max((self.gp.y_sample.min() - y_trial), 0)

            exp_imp = _compute_ic(self.gp, x_trial)

            if y_trial < self.y_local:
                self.x_local = x_trial
                self.y_local = y_trial

            if exp_imp != 0:
                if actual_imp / exp_imp >= 1:
                    self._set_step_size(True)
                    
                elif actual_imp / exp_imp == 0:
                    self._set_step_size(False)

            else:
                self._set_step_size(False)


        if self.param['mode'] == 'gcego_restart':

            exp_imp = _compute_ic(self.gp, x_trial)

            if exp_imp < 0.001:
                d = self.x_local.size
                x_sample = lhs_sampling(2 * (d + 1), d, obj_func.bounds)
                y_sample = np.zeros((2 * (d + 1), 1))

                for i, xi in enumerate(x_sample):
                    y_sample[i] = obj_func.evaluate(xi)

                self.gp.__init__(x_sample, y_sample)

                self.x_local = x_sample[y_sample.argmin()]
                self.y_local = y_sample.min()

            if y_trial <= self.y_local - self.enforcing_func(self.step_size):
                self.x_local = x_trial
                self.y_local = y_trial
                self._set_step_size(True)

            else:
                self._set_step_size(False)
                
                
        if self.param['mode'] == 'trike_restart':

            actual_imp = max((self.gp.y_sample.min() - y_trial), 0)

            exp_imp = _compute_ic(self.gp, x_trial)

            if y_trial < self.y_local:
                self.x_local = x_trial
                self.y_local = y_trial

            if exp_imp < 0.001:
                d = self.x_local.size
                x_sample = lhs_sampling(2 * (d + 1), d, obj_func.bounds)
                y_sample = np.zeros((2 * (d + 1), 1))

                for i, xi in enumerate(x_sample):
                    y_sample[i] = self.gp.f.evaluate(xi)

                self.gp.restart(x_sample, y_sample)

                self.x_local = x_sample[y_sample.argmin()]
                self.y_local = y_sample.min()

            if exp_imp != 0:
                if actual_imp/exp_imp >= 1:
                    self._set_step_size(True)
                    
                elif actual_imp/exp_imp == 0:
                    self._set_step_size(False)

            else:
                self._set_step_size(False)

        return self.step_size
