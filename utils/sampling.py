#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:30:47 2018

@author: a-scotto
"""

import numpy as np
from numpy.random import randint, rand

def _get_interval_length(x_limits, n):

    lengths = []
    k = x_limits.shape[0]

    for i in range(k):

        if x_limits[i, 1] - x_limits[i, 0] <= 0:
            raise ValueError("Limits for axis {} are inconsistent. Upper limit (={}) should be strictly greater than lower limit (={})".format(i, x_limits[i, 1], x_limits[i, 0]))

        lengths.append((x_limits[i, 1] - x_limits[i, 0]) / n)

    return lengths

def lhs_sampling(n, k, bounds):

    x_limits = np.zeros((k, 2))
    
    if len(bounds) != k:
        raise ValueError('Bounds list length (len={}) must be equal to variables number (k={}).'.format(len(bounds), k))
    
    for i, b in enumerate(bounds):
        x_limits[i, 0] = b[0]
        x_limits[i, 1] = b[1]

    A = [[i for i in range(n)] for j in range(k)]
    P = []

    n_remaining = n

    for i in range(n):
        p = []

        for j in range(k):
            r = randint(n_remaining)
            p.append(A[j][r])
            A[j].remove(A[j][r])

        n_remaining = n_remaining - 1
        P.append(p)

    lengths = _get_interval_length(x_limits, n)

    X = np.zeros((n, k))

    for i in range(n):

        for j in range(k):

            interval_no = P[i][j]

            a = x_limits[j, 0] + interval_no * lengths[j]

            b = a + lengths[j]

            X[i, j] = (b - a) * rand() + a

    return X


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    n = 5

    bounds = [(0, 1), (0, 1)]
    sample = lhs_sampling(n, 2, bounds)
    
    fig = plt.figure(figsize=(16, 16))
    
    xcoord = np.linspace(bounds[0][0], bounds[0][1], n + 1)
    for xc in xcoord:
        plt.axvline(x = xc, c='black', lw=0.5)

    ycoord = np.linspace(bounds[1][0], bounds[1][1], n + 1)
    for yc in ycoord:
        plt.axhline(y = yc, c='black', lw=0.5)
        
    plt.axis([bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]])
    plt.title('Latin Hypercube Sampling example')
    plt.xlabel('Decision variable 1')
    plt.ylabel('Decision variable 2')
    plt.plot(sample[:, 0], sample[:, 1], 'r+', mew=2.5, ms=20.)