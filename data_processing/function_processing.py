#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:55:16 2017

@author: a.scotto
"""

import os
import pickle
from numpy import zeros
from numpy.random import rand
from scipy.optimize import basinhopping
from functions_bib.function_generator import get_func

class Function(object):

    def __init__(self, obj_func, function_number=0, var_number=0, arrival_dim=0, modification_type=0):
        
        self.obj_func = obj_func
        
        self.func_id = (function_number, 
                        var_number, 
                        arrival_dim, 
                        modification_type)
        
        self.x_opt = get_minimizer(obj_func, self.func_id)
        self.bounds = None
                
        self.eval_count = 0
        self.evals = []


    def set_bounds(self, size):
        
        bounds = []
        
        for i, x_opt_i in enumerate(self.x_opt):
            bounds.append((x_opt_i - size, x_opt_i + size))
        
        self.bounds = bounds


    def evaluate(self, x):
        y = float(self.obj_func(x))
        self.eval_count += 1
        self.evals.append(y)
        return y


    def reset(self):
        self.eval_count = 0
        self.evals = []


def extract_test_set(file_name):

    function_set = []

    if not os.listdir("test_files"):
        raise FileNotFoundError("The 'test_files' directory contains no test files. \n")


    with open("test_files/" + file_name, 'r') as file:

        L = file.read().split("\n")

        for l in L:

            try:
                func_number, d, m, modes = l.split(" ")
                func_number, d, m = int(func_number), int(d), int(m)

                for modification_type in modes.split(","):
                    modification_type = int(modification_type)
                    f = get_func(func_number, d, m, modification_type)
                    function_set.append(Function(f, func_number, d, m, modification_type))

            except:
                pass

    return function_set


def get_minimizer(obj_func, func_id):
    
    func_number, d, m, modification_type = func_id

    with open("data_processing/func_data", 'rb') as func_data_file:
        p = pickle.Unpickler(func_data_file)
        func_data = p.load()

    if (func_number, d, m, modification_type) not in func_data.keys():
        print("New problem tested ID={}, d={}, m={}, type={}".format(func_number, d, m, modification_type))
        x_opt = optimize(obj_func, d)
        func_data[func_number, d, m, modification_type] = x_opt

        with open("data_processing/func_data", 'wb') as func_data_file:
            p = pickle.Pickler(func_data_file)
            p.dump(func_data)
            print("Saved.")

    else:
        x_opt = func_data[func_number, d, m, modification_type]

    return x_opt


def optimize(f, d):

    print("Optimization in progress ", end="")

    x_opt = zeros((d))
    f_opt = 1e100

    radius = [0.01, 0.1, 1, 5, 10, 25, 50, 100]

    for r in radius:
        x_0 = 0.125 * r * (rand(d) - 1 / 2)
        res = basinhopping(lambda x : f(x), x_0)
        print(".", end="")
        if res.fun < f_opt:
            x_opt = res.x

    print("  Terminated.")

    return x_opt


def get_test_overview(file_name):

    function_set = extract_test_set(file_name)
    dim = []

    for f in function_set:
        func_number, d, m, modification_type = f.func_id
        dim.append(d)

    dim.sort()
    temp = 0
    t = 0

    print("Test set:", len(dim), " problems.")

    for d in dim:
        if d != temp:
            print(" ({})".format(t), end="")
            d_temp = str(d)
            if len(d_temp) == 1:
                print("\n ", d, ": |", end="")
            else:
                print("\n", d, ": |", end="")
            temp = d
            t = 1

        else:
            t += 1
            print("|", end="")

    print(" ({})".format(t), end="")

