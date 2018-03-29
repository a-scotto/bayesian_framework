#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:16:21 2017
@author: a.scotto
"""
import os
import pickle
import numpy as np


def save_data(fileName, data, dimensions, solvers):

    # Create 'results' directory if not yet created
    if "results" not in os.listdir():
        os.mkdir("results")

    # Save file in the 'results' directory with name fileName
    with open("results/" + fileName + ".txt", 'wb') as myFile:
        p = pickle.Pickler(myFile)
        p.dump([data, dimensions, solvers])


def load_data(fileName):

    # Open fileName.txt in the 'results' directory and store data

    with open("results/" + fileName, 'rb') as myFile:
        p = pickle.Unpickler(myFile)
        L = p.load()
        data, dimensions, solvers = L

    return data, dimensions, solvers


def shape_data(L):

    # Get the number of problems and number of solvers
    n_prob = len(L)
    n_solv = len(L[0])

    # Initialize variables
    n_it = 0
    data = []

    # Go through all the problems
    for problem in L:

        # For each problem go through all the solvers
        for solver in problem:

            # Fill in list results for solver s and problem p
            data.append(solver)

            # If number of iteration greater than the current one, increase it
            if solver.size > n_it:
                n_it = solver.size

    # Create the matrix where results are aimed to be stored
    shaped_data = np.zeros((n_prob, n_solv, n_it))

    # Go through the list of results
    for i, r in enumerate(data):

        # Complete the r vector with the minimum value to adjust sizes

        temp = r.min() * np.ones((n_it - r.size, 1))
        r = r.reshape((r.size, 1))
        r = np.vstack((r, temp))

        # Get p and s from the flat results
        p = i // n_solv
        s = i % n_solv

        # Fill in the results amtrix
        shaped_data[p, s, :] = r.reshape((n_it))

    return shaped_data
