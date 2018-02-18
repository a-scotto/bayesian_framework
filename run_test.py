#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:25:36 2017
@author: a.scotto
"""

import time
from numpy import round
from utils.sampling import lhs_sampling
from utils.parameter import Parameter
from data_processing.function_processing import extract_test_set
from data_processing.post_processing import saveData
from optimizer.bayesian_optim import BayesianOptimizer

def main(i=0):

    # FILES PARAMETERS
    testFileName = 'fullDimension'
    box_size = 50.0
    tolerance = 1e-5
    
    # SOLVER INSTANCIATION
    egoModeTested = ['elgowlm', 'elgo', 'gcego']
    solvers = []

    for mode in egoModeTested:
        solvers.append(BayesianOptimizer(Parameter({'mode' : mode, 'verbosity' : False})))

    # DATA INITIALIZATION
    functionSet = extract_test_set(testFileName + '.txt')
    saveFileName = '_'.join([testFileName,
                             '&'.join(egoModeTested).upper(),
                             solvers[0].param['ic'],
                             'b=' + str(box_size),
                             'tol=' + str(tolerance),
                             'run=' + str(i)])

    results, dimensions = [], []

    for obj_func in functionSet:
        
        print('* Problem :', obj_func.func_id[0], 
              '| Dimension :', obj_func.func_id[1], 
              '| Type :', obj_func.func_id[3])
        
        d = obj_func.func_id[1]
        
        if tolerance == 0:
            max_budget = max(25 * (d + 1), 500)
        else:
            max_budget = 50 * (d + 1)

        obj_func.set_bounds(box_size)
        x_sample = lhs_sampling(2 * (d + 1), d, obj_func.bounds)
        
        dimensions.append(d)
        p = []

        for s in solvers:
            t_start = time.time()
            evals, cost, x_opt = s.run(obj_func, x_sample, max_budget, tolerance)
            p.append(evals)
            obj_func.reset()
            
            print('    - ' + s.param['mode'] + ' terminated in', round(time.time() - t_start, 2), 's and', cost, 'evaluations || Value : ', float(min(evals)))

        results.append(p)
        saveData(saveFileName, results, dimensions, egoModeTested)
        print('{}/{} problem(s) solved. \n'.format(len(results), len(functionSet)))

    saveData(saveFileName, results, dimensions, egoModeTested)

if __name__ == '__main__':

    for i in range(10):
        main(i)