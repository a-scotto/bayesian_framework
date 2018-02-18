#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:15:50 2018

@author: a-scotto
"""

class Parameter(object):
    
    param_list = ['decr_fact', 
                  'incr_fact', 
                  'init_step_size',
                  'verbosity',
                  'kernel',
                  'ic',
                  'mode']
    
    def __init__(self, parameters=None):
        
        self.param = self._set_default_values()
        
        if parameters is not None:
            self._set_param(parameters)
        
    
    def _set_default_values(self):
        
        default_value = {'decr_fact' : 0.5,
                         'incr_fact' : 2.0,
                         'init_step_size' : 1.0,
                         'verbosity' : False,
                         'kernel' : 'sq_exp',
                         'ic' : 'exp_imp',
                         'mode' : 'elgo'}
    
        return default_value
    
    def _set_param(self, parameters):
        
        for par, param_value in parameters.items():
            
            if par not in self.param_list:
                raise ValueError('{} not a parameter. Please refer to parameters list.'.format(par))
            
            else:
                self._param_checks(par, param_value)
                self.param[par] = param_value


    def _param_checks(self, par, param_value):
        
        if par == 'decr_fact':            
            if param_value <= 0 or param_value >= 1:
                raise ValueError('{} must be in ]0, 1[ interval but received {}.'.format(par, param_value))
        
        elif par == 'incr_fact':            
            if param_value <= 1:
                raise ValueError('{} must be > 1 but received {}.'.format(par, param_value))
        
        elif par == 'init_step_size':            
            if param_value <= 0:
                raise ValueError('{} must be > 0 but received {}.'.format(par, param_value))
        
        elif par == 'verbosity':            
            if type(param_value) != bool:
                raise ValueError('{} must be a boolean but received {}.'.format(par, param_value))

        elif par == 'kernel':            
            if param_value not in ['sq_exp', 'ornstein']:
                raise ValueError('{} is not defined. See kernel list.'.format(param_value))
        
        elif par == 'ic':            
            if param_value not in ['exp_imp', 'wb2']:
                raise ValueError('{} is not defined. See infill criteria list.'.format(param_value))
        
        elif par == 'mode':            
            if param_value not in ['ego', 'elgo', 'gcego', 'gcego_restart', 'trike', 'trike_restart', 'elgowlm']:
                raise ValueError('{} is not defined. See mode list.'.format(param_value))
        