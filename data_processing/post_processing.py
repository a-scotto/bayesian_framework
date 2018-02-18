#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:16:21 2017
@author: a.scotto
"""
import os
import pickle
import matplotlib.pyplot as plt
from numpy import ones, zeros, sort, linspace, vstack, abs, Inf, isinf


def saveData(fileName, data, dimensions, solverNames):

    # Create 'results' directory if not yet created
    if "results" not in os.listdir():
        os.mkdir("results")

    # Save file in the 'results' directory with name fileName
    with open("results/" + fileName + ".txt", 'wb') as myFile:
        p = pickle.Pickler(myFile)
        p.dump([data, dimensions, solverNames])


def loadData(fileName):

    # Open fileName.txt in the 'results' directory and store data

    with open("results/" + fileName, 'rb') as myFile:
        p = pickle.Unpickler(myFile)
        L = p.load()
        data, dimensions, solverNames = L

    return data, dimensions, solverNames


def prepareData(L):

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
    preparedData = zeros((n_prob, n_solv, n_it))

    # Go through the list of results
    for i, r in enumerate(data):

        # Complete the r vector with the minimum value to adjust sizes

        temp = r.min() * ones((n_it - r.size, 1))
        r = r.reshape((r.size, 1))
        r = vstack((r, temp))

        # Get p and s from the flat results
        p = i // n_solv
        s = i % n_solv

        # Fill in the results amtrix
        preparedData[p, s, :] = r.reshape((n_it))

    return preparedData


def mergeSolvers(*data):

    agg_results = []
    agg_solv = []

    if data is []:
        raise ValueError("No file provided.")

    for d in data:
        res, dim, solv = loadData(d)
        agg_solv.extend(solv)
        agg_results.append(res)

    n_prob = len(agg_results[0])

    for res in agg_results:
        if len(res) != n_prob:
            raise ValueError("Number of problem tested do not match between different files.")

    res = agg_results[0]

    for i in range(n_prob):
        for j in range(1, len(agg_results)):
            res[i].extend(agg_results[j][i])

    return res, dim, agg_solv


def getDataProfile(data, dimensions, solverNames, alpha, save=""):

    # Transform list of results into matrix
    data = prepareData(data)
    n_prob, n_solv, n_it = data.shape

    maxBudgetReached = 0
    fig = plt.figure(figsize=(10, 6))

    for s in range(n_solv):

        budgetRequired = []
        problemSolved = []

        for p in range(n_prob):

           fL = data[p, :, :].min()

           f0 = data[p, s, :2*(dimensions[p] + 1)].min()

           cutoff = alpha * f0 + (1 - alpha) * fL

           for k in range(min(100 * (dimensions[p] + 1), n_it)):

                if data[p, s, k] < cutoff:
                    budgetRequired.append(k / (dimensions[p] + 1))
                    problemSolved.append(100 * len(budgetRequired) / n_prob)
                    break

        budgetRequired.append(100)
        if len(problemSolved) == 0:
            problemSolved.append(0)
        else:
            problemSolved.append(problemSolved[-1])

        budgetRequired = sort(budgetRequired)

        if budgetRequired[-2] > maxBudgetReached:
            maxBudgetReached = budgetRequired[-2]

        plt.step(budgetRequired, problemSolved)

    plt.title("Data Profile with {} problems - Accuracy {} = {}".format(n_prob, chr(945), alpha))
    plt.xlabel('Simplex Gradient [Budget / (d + 1)]')
    plt.ylabel('Percentage of problem solved')
    plt.axis([0, maxBudgetReached, 0, 100])
    plt.legend(solverNames)
    plt.grid()
    plt.show()

    # Save file as .png if a file name was specified in the 'save' arguments
    if save != "":
        fig.savefig("results/DataProfile Acc=" + str(alpha) + " " + save + ".png", bbox_inches='tight')


def getPerformanceProfile(data, dimensions, solverNames, alpha, save=""):

    # Transform list of results into matrix
    data = prepareData(data)
    n_prob, n_solv, n_it = data.shape

    # Initialize the performance ratios matrix
    rps = zeros((n_solv, n_prob))

    # Go through all problems
    for p in range(n_prob):

        # Get for each problem the minimum value reached among all solvers
        fL = data[p, :, :].min()

        # Define the cutoff for convergence test
        cutoff = fL + alpha*abs(fL) + alpha

        # Initialize high value for the reference performance value
        tps = Inf * ones((n_solv))

        # Definition of the reference performance value for problem p
        for s in range(n_solv):

            for k in range(2 * (dimensions[p] + 1), n_it):

                if data[p, s, k] <= cutoff:
                    tps[s] = k
                    break

        rps[:, p] = tps / tps.min()

    # Plot the results
    fig = plt.figure(figsize=(12, 6))

    tau, rho, tau_m = [], [], []

    # Computation of performance ratios of each solver s for current problem p
    for s in range(n_solv):

        rp = sort(rps[s, :])

        tau_s, rho_s = [], []

        for p in range(n_prob):

            if not isinf(rp[p]):

                tau_s.append(rp[p])
                rho_s.append(p/n_prob)

        tau_m.append(max(tau_s))

        tau.append(tau_s)
        rho.append(rho_s)

    for s, tau_s in enumerate(tau):
        tau_s.append(max(tau_m))
        rho[s].append(rho[s][-1])
        plt.step(tau_s, rho[s])


    plt.title("Performance Profile over {} problems - Accuracy {} = {}".format(n_prob, chr(945), alpha))
    plt.xlabel("Peformance Ratio")
    plt.ylabel("Percentage of problem solved")
    plt.legend(solverNames)
    plt.xlim(xmax=max(tau_m))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    # Save file as .png if a file name was specified in the 'save' arguments
    if save != "":
        fig.savefig("results/PerformanceProfile Acc=" + str(alpha) + " " + save + ".png", bbox_inches='tight')

if __name__ == "__main__":

    os.chdir("..")

    data, s = loadData("AS_TestFile_New_Test_DFO_EGO-INTR_EGO-DFO_TRIKE")

    getDataProfile(data, s, 0.001)
    getPerformanceProfile(data, s, 0.1)