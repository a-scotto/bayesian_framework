#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:17:20 2018

@author: a-scotto
"""

import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from utils.ic import _compute_ic
from gauss_process.gauss_process import GaussianProcess

plt.rc('text', usetex=True)
fs = 25
plt.rcParams.update({'font.size': fs})
plt.ion()

def f(x):
    return 20 * np.sin(x) * np.exp(-(x / 10)**2) + (x / 5 - 1)**2 + np.exp(x / 5)

a = 20
n_sample = 5
crit = 'exp_imp'
x = np.linspace(-a, a, n_sample).reshape((n_sample, 1))
y = f(x).reshape((n_sample, 1))

gp = GaussianProcess(x, y, kernel='sq_exp')

x_plot = np.linspace(-a, a, 500)
infer = np.zeros(x_plot.shape)
ic = np.zeros(x_plot.shape)

fig = plt.figure(figsize=(10, 9))

cond = [lg.cond(gp.K)]
theta = [float(gp.cov_matrix.theta)]

for i in range(30):
    for i in range(x_plot.size):
        infer[i] = gp.infer_value(x_plot[i])
        ic[i] = _compute_ic(gp, x_plot[i])
    
    fig.clf()
    plt.subplot(211)
    plt.title("Sample size = {} ~~\\vline~~ $\\theta$ = {} ~~\\vline~~ Cond(K) = {}".format(gp.y_sample.size, 
              round(float(gp.cov_matrix.theta), 2), 
              round(lg.cond(gp.K), 2)))
    plt.plot(x_plot, f(x_plot), '--')
    plt.plot(x_plot, infer)
    plt.plot(gp.x_sample, gp.y_sample, 'r*')
    plt.plot(gp.x_sample[-1], gp.y_sample[-1], 'go')
    plt.xlim(-a, a)
    
    plt.subplot(212)    
    plt.plot(x_plot, ic)
    x_next = gp.enrich_model([(-a, a)], crit)
    plt.plot(x_next, _compute_ic(gp, x_next, crit), 'r*')
    plt.xlim(-a, a)
    fig.show()
    
    plt.pause(0.01)
    input("Press any key to iterate. ")
    gp.add_sample(x_next, f(x_next))
