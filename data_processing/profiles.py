#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Mar  9 11:08:14 2018

@author: a-scotto
'''
import numpy
import numpy as np
import matplotlib.pyplot as plt
from data_processing.post_processing import shape_data

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})


def data_profile(data, dimensions, solvers, alpha, save=''):

    # Transform list of results into matrix
    data = shape_data(data)
    n_prob, n_solv, n_it = data.shape
    max_budget = 100

    for s in range(n_solv):
        required_budget = [2]
        solved_prob = [0]
        for p in range(n_prob):
            f_best = data[p, :, :].min()
            f0 = data[p, s, :2*(dimensions[p] + 1)].min()
            cutoff = alpha * f0 + (1 - alpha) * f_best
            for k in range(min(max_budget * (dimensions[p] + 1), n_it)):
                if data[p, s, k] < cutoff:
                    required_budget.append(k / (dimensions[p] + 1))
                    solved_prob.append(solved_prob[-1] + 100 / n_prob)
                    break

        # if len(solved_prob) != n_prob + 1:
        required_budget.append(max_budget)
        solved_prob.append(solved_prob[-1])

        required_budget = np.sort(required_budget)

        plt.step(required_budget, solved_prob, '-s', where='post', lw=2, ms=6, mfc=None, mec='k')

    plt.title('Data Profile, $\\alpha = 10^{' + str(int(np.log10(alpha))) + '}$')
    plt.xlabel('Simplex Gradient [Budget / (d + 1)]')
    plt.ylabel('Percentage of problem solved')
    plt.axis([2, max_budget, 0, 100])
    plt.legend(solvers)
    plt.grid()
    plt.show()

    # Save file as .png if a file name was specified in the 'save' arguments
    if save != '':
        fig.savefig('results/DP_{}_alpha=1e{}.pdf'.format(save, int(np.log10(alpha))))


def performance_profile(data, dimensions, solvers, alpha, save=''):

    # Transform list of results into matrix
    data = shape_data(data)
    n_prob, n_solv, n_it = data.shape
    max_budget = 100

    # Initialize the performance ratios matrix
    rps = np.zeros((n_solv, n_prob))

    # Go through all problems
    for p in range(n_prob):
        # Get for each problem the minimum value reached among all solvers
        f_best = data[p, :, :].min()

        # Define the cutoff for convergence test
        cutoff = f_best + alpha*abs(f_best)
        # Initialize high value for the reference performance value
        tps = np.Inf * np.ones((n_solv))

        # Definition of the reference performance value for problem p
        for s in range(n_solv):
            for k in range(min(max_budget * (dimensions[p] + 1), n_it)):
                if data[p, s, k] <= cutoff:
                    tps[s] = k
                    break

            print(solvers[s], tps)

        rps[:, p] = tps / tps.min()

    print(rps)

    # Plot the results
    tau, rho, tau_m = [], [], []

    # Computation of performance ratios of each solver s for current problem p
    for s in range(n_solv):
        rp = np.sort(rps[s, :])
        tau_s, rho_s = [], []

        for p in range(n_prob):
            if not np.isinf(rp[p]):
                tau_s.append(rp[p])
                rho_s.append(100 * (p + 1) / n_prob)

        tau_m.append(max(tau_s))
        tau.append(tau_s)
        rho.append(rho_s)

        print(tau_s, rho_s)

    for s, tau_s in enumerate(tau):
        tau_s.append(max(tau_m))
        rho[s].append(rho[s][-1])
        plt.step(tau_s, rho[s], '-s', where='post', lw=2, ms=6, mfc=None, mec='k')

    plt.title('Performance Profile, $\\alpha = 10^{' + str(int(np.log10(alpha))) + '}$')
    plt.xlabel('Peformance Ratio')
    plt.ylabel('Percentage of problem solved')
    plt.legend(solvers)
    plt.xlim(1, max(tau_m))
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

    # Save file as .png if a file name was specified in the 'save' arguments
    if save != '':
        fig.savefig('results/PP_{}_alpha=1e{}.pdf'.format(save, int(np.log10(alpha))))