#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:05:13 2018

@author: a-scotto
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution

np.seterr(all='ignore')

def _compute_ic(gaussian_proc, x, crit='exp_imp'):

    y = gaussian_proc.infer_value(x)
    s = np.sqrt(gaussian_proc.estimate_mse(x))
    y_min = gaussian_proc.y_sample.min()

    if crit == 'exp_imp':
        if s == 0.:
            ic = 0.
        elif s > 0:
            y_red = (y_min - y) / s
            ic = (y_min - y) * norm.cdf(y_red) + s * norm.pdf(y_red)
    
    elif crit == 'wb2':
        if s == 0.:
            ic = - y
        elif s > 0.:
            y_red = (y_min - y) / s
            ic = - y + (y_min - y) * norm.cdf(y_red) + s * norm.pdf(y_red)

    else:
        raise ValueError('Infill criterion "{}" not defined.'.format(crit))

    return float(ic)


def maximize_ic(gaussian_proc, bounds, crit='exp_imp'):

    optim_res = differential_evolution(lambda x: - _compute_ic(gaussian_proc, x, crit), bounds, maxiter=5)

    xEGO = optim_res.x

    return xEGO