#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:47:45 2017
@author: a.scotto
"""

from scipy.linalg import norm
from numpy.random import seed, randn
from numpy import arctan, pi, cos, sin, exp, sqrt, log
from numpy import isnan, isinf, any, zeros, ones, sum, prod, arange, array, seterr

seterr(all='ignore')


def getTestFunction(functionNumber, d, m, modificationType):

    v = [4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2]
    y1 = [1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39]
    y2 = [1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2]
    y3 = [3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4, 9.744e3, 8.261e3, 7.03e3, 6.005e3, 5.147e3, 4.427e3, 3.82e3, 3.307e3, 2.872e3]
    y4 = [8.44e-1, 9.08e-1, 9.32e-1, 9.36e-1, 9.25e-1, 9.08e-1, 8.81e-1, 8.5e-1, 8.18e-1, 7.84e-1, 7.51e-1, 7.18e-1, 6.85e-1, 6.58e-1, 6.28e-1, 6.03e-1, 5.8e-1, 5.58e-1, 5.38e-1, 5.22e-1, 5.06e-1, 4.9e-1, 4.78e-1, 4.67e-1, 4.57e-1, 4.48e-1, 4.38e-1, 4.31e-1, 4.24e-1, 4.2e-1, 4.14e-1, 4.11e-1, 4.06e-1]
    y5 = [1.366, 1.191, 1.112, 1.013, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1, 7.86e-1, 7.25e-1, 7.46e-1, 6.79e-1, 6.08e-1, 6.55e-1, 6.16e-1, 6.06e-1, 6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1, 6.49e-1, 6.94e-1, 6.44e-1, 6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1, 4.95e-1, 5.0e-1, 4.23e-1, 3.95e-1, 3.75e-1, 3.72e-1, 3.91e-1, 3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, 5.23e-1, 5.62e-1, 6.07e-1, 6.53e-1, 6.72e-1, 7.08e-1, 6.33e-1, 6.68e-1, 6.45e-1, 6.32e-1, 5.91e-1, 5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1, 7.29e-1, 7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1, 1.62e-1, 9.8e-2, 5.4e-2]

    def modifiedFunctionCreator(modificationType):

        def modifyFunction(f):

            def modifiedFunction(x):

                f_x = f(x)

                # Ensure no 'NaNs' or 'Inf' are contained in f_x vector

                if any(isnan(f_x)) or any(isinf(f_x)):
                    incorrectValues = isnan(f_x) + isinf(f_x)
                    for i, b in enumerate(incorrectValues):
                        if b:
                            f_x[i] = 1e15

                # Hard modification
                if modificationType == 1:
                    y = norm(f_x, 1)

                # Smooth modification
                elif modificationType == 2:
                    y =  norm(f_x, 2)

                # White noise modification
                elif modificationType == 3:
                    sigma = 10**(-3)
                    seed(0)
                    u = 1 + sigma * (-ones((m, 1)) + 2 * randn(m, 1))
                    y =  norm(f_x * u, 2)

                # Coherent noise modification
                elif modificationType == 4:
                    sigma = 10**(-3)
                    phi = 0.9 * sin(100 * norm(x, 1)) * cos(100 * norm(x, 2)) + 0.1 * cos(norm(x, 2))
                    phi = phi * (4 * phi**2 - 3)
                    y = (1 + sigma * phi) *  norm(f_x, 2)

                return min(y, 1e15)

            return modifiedFunction

        return modifyFunction

    modificator = modifiedFunctionCreator(modificationType)

    if functionNumber == 1:  # LINEAR function - FULL RANK

        @modificator
        def f(x):
            f_x = zeros((m, 1))

            t = 2 * sum(x) / m + 1

            for i in range(m):
                f_x[i] = -t
                if i < x.size:
                    f_x[i] = -t + x[i]

            return f_x


    elif functionNumber == 2:  # LINEAR function - RANK 1

        @modificator
        def f(x):
            f_x = zeros((m, 1))
            i = arange(0, x.size) + 1
            t = sum(i * x)

            for i in range(m):
                f_x[i] = (i + 1) * t - 1

            return f_x


    elif functionNumber == 3:  # LINEAR function - RANK 1 with zero columns and rows

        @modificator
        def f(x):
            f_x = zeros((m, 1))
            t = 0

            for i in range(1, x.size - 1):
                t = t + (i + 1) * x[i]

            for i in range(m - 1):
                f_x[i] = i * t - 1

            f_x[-1] = -1

            return f_x


    elif functionNumber == 4:  # ROSENBROCK

        @modificator
        def f(x):
            f_x = zeros((2, 1))

            f_x[0] = 10 * (x[1] - x[0]**2)
            f_x[1] = 1 - x[0]

            return f_x


    elif functionNumber == 5:  # HELICAL VALLEY function

        @modificator
        def f(x):
            f_x = zeros((3, 1))

            if x[0] > 0:
                t = arctan(x[1] / x[0]) / pi / 2
            elif x[0] < 0:
                t = arctan(x[1] / x[0]) / pi / 2 + 1 / 2
            else:
                t = 1 / 4

            r = sqrt(x[0]**2 + x[1]**2)

            f_x[0] = 10 * (x[2] - 10 * t)
            f_x[1] = 10 * (r - 1)
            f_x[2] = x[2]

            return f_x


    elif functionNumber == 6:  # POWELL SINGULAR function

        @modificator
        def f(x):
            f_x = zeros((4, 1))

            f_x[0] = x[0] + 10 * x[1]
            f_x[1] = sqrt(5) * (x[2] - x[3])
            f_x[2] = (x[1] - 2 * x[2])**2
            f_x[3] = sqrt(10) * (x[0] - x[3])**2

            return f_x


    elif functionNumber == 7:  # FREUDENSTEIN & ROTH function

        @modificator
        def f(x):
            f_x = zeros((2, 1))

            f_x[0] = -13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
            f_x[1] = -29 + x[0] + ((1 + x[1]) * x[1] - 14) * x[1]

            return f_x


    elif functionNumber == 8:  # BARD function

        @modificator
        def f(x):
            f_x = zeros((15, 1))

            for i in range(15):
                t1 = i + 1
                t2 = 16 - t1
                t3 = t1

                if i > 7:
                    t3 = t2

                f_x[i] = y1[i] - (x[0] + t1 / (x[1] * t2 + x[2] * t3))

            return f_x


    elif functionNumber == 9:  # KOWALIK and OSBORNE function
        @modificator
        def f(x):
            f_x = zeros((11, 1))

            for i in range(11):
                t1 = v[i] * (v[i] + x[1])
                t2 = v[i] * (v[i] + x[2]) + x[3]

                f_x[i] = y2[i] - x[0] * t1 / t2

            return f_x


    elif functionNumber == 10:  # MEYER function
        @modificator
        def f(x):
            f_x = zeros((16, 1))

            for i in range(16):
                t = 5 * (i + 1) + 45 + x[2]
                t1 = x[1] / t
                t2 = min(exp(t1), 1e100)
                f_x[i] = x[0] * t2 - y3[i]

            return f_x


    elif functionNumber == 11:  # WATSON function
        @modificator
        def f(x):
            f_x = zeros((31, 1))

            for i in range(29):
                d = (i + 1) / 29
                s1 = 0
                dx = 1
                for j in range(1, x.size):
                    s1 += j * dx * x[j]
                    dx = d * dx

                s2 = 0
                dx = 1
                for j in range(x.size):
                    s2 += dx * x[j]
                    dx = d * dx

                f_x[i] = s1 - s2**2 - 1

            f_x[29] = x[0]
            f_x[30] = x[1] - x[0]**2 - 1

            return f_x


    elif functionNumber == 12:  # BOx 3-DIMENSIONAL function
        @modificator
        def f(x):
            f_x = zeros((m, 1))

            for i in range(m):
                t = (i + 1) / 10
                f_x[i] = exp(-t * x[0]) - exp(-t * x[1]) + (exp(-(i + 1)) - exp(-t)) * x[2]

            return f_x


    elif functionNumber == 13:  # JENNRICH & SAMPSON function
        @modificator
        def f(x):
            f_x = zeros((m, 1))

            for i in range(m):
                t = i + 1
                f_x[i] = 2 + 2 * t - exp(t * x[0]) - exp(t * x[1])

            return f_x


    elif functionNumber == 14:  # BROWN & DENNIS function
        @modificator
        def f(x):
            f_x = zeros((m, 1))

            for i in range(m):
                t = (i + 1) / 5
                t1 = x[0] + t * x[1] - exp(t)
                t2 = x[2] + sin(t) * x[3] - cos(t)
                f_x[i] = t1**2 + t2**2

            return f_x


    elif functionNumber == 15:  # CHEBYQUAD function
        @modificator
        def f(x):
            f_x = zeros((m, 1))

            for j in range(x.size):
                t1 = 1
                t2 = 2 * x[j] - 1
                t = 2 * t2
                for i in range(m):
                    f_x[i] += t2
                    th = t * t2 - t1
                    t1 = t2
                    t2 = th

            inv = -1
            for i in range(m):
                f_x[i] /= x.size

                if inv > 0:
                    f_x[i] += 1 / ((i + 1)**2 - 1)

                inv *= -1

            return f_x


    elif functionNumber == 16:  # BROWN almort-linear function
        @modificator
        def f(x):
            f_x = zeros(x.shape)

            s = sum(x) - (x.size + 1)
            p = prod(x)

            for i in range(x.size - 1):
                f_x[i] = x[i] + s

            f_x[-1] = p - 1

            return f_x


    elif functionNumber == 17:  # OSBORNE 1 function
        @modificator
        def f(x):
            f_x = zeros((33, 1))

            for i in range(33):
                t = 10 * i
                t1 = exp(-x[3] * t)
                t2 = exp(-x[4] * t)
                f_x[i] = y4[i] - (x[0] + x[1] * t1 + x[2] * t2)

            return f_x


    elif functionNumber == 18:  # OSBORNE 2 function
        @modificator
        def f(x):
            f_x = zeros((65, 1))

            for i in range(65):
                t = i / 10
                t1 = exp(-x[4] * t)
                t2 = exp(-x[5] * (t - x[8])**2)
                t3 = exp(-x[6] * (t - x[9])**2)
                t4 = exp(-x[7] * (t - x[10])**2)

                f_x[i] = y5[i] - (x[0] * t1 + x[1] * t2 + x[2] * t3 + x[3] * t4)

            return f_x


    elif functionNumber == 19:  # BDQRTIC function
        @modificator
        def f(x):
            f_x = zeros(((x.size - 4) * 2, 1))

            for i in range(x.size - 4):
                f_x[i] = -4 * x[i] + 3
                f_x[x.size - 4 + i] = x[i]**2 + 2 * x[i + 1]**2 + 3 * x[i + 2]**2 + 4 * x[i + 3]**2 + 5 * x[-1]**2

            return f_x


    elif functionNumber == 20:  # CUBE
        @modificator
        def f(x):
            f_x = zeros(x.shape)

            f_x[0] = (x[0] - 1)

            for i in range(1, x.size):
                f_x[i] = 10 * (x[i] - x[i - 1]**3)

            return f_x


    elif functionNumber == 21:  # MANCINO function
        @modificator
        def f(x):
            f_x = zeros(x.shape)

            for i in range(x.size):
                s = 0
                for j in range(x.size):
                    v = sqrt(x[i]**2 + (i + 1) / (j + 1))
                    s += v * (sin(log(v)))**5 + cos(log(v))**5

                f_x[i] = 1400 * x[i] + (i + 1 - 50)**3 + s

            return f_x


    elif functionNumber == 22:  # HEART8LS function
        @modificator
        def f(x):
            f_x = zeros(x.shape)

            f_x[0] = x[0] + x[1] + 0.69
            f_x[1] = x[2] + x[3] + 0.044
            f_x[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] + 1.57
            f_x[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[4] + 1.31
            f_x[4] = x[0] * (x[4]**2 - x[6]**2) - 2.0 * x[2] * x[4] * x[6] + x[1] * (x[5]**2 - x[7]**2) - 2.0 * x[3] * x[5] * x[7] + 2.65
            f_x[5] = x[2] * (x[4]**2 - x[6]**2) + 2.0 * x[0] * x[4] * x[6] + x[3] * (x[5]**2 - x[7]**2) + 2.0 * x[1] * x[5] * x[7] - 2.0
            f_x[6] = x[0] * x[4] * (x[4]**2 - 3.0 * x[6]**2) + x[2] * x[6] * (x[6]**2 - 3.0 * x[4]**2) + x[1] * x[5] * (x[5]**2 - 3.0 * x[7]**2) + x[3] * x[7] * (x[7]**2 - 3.0 * x[5]**2) + 12.6
            f_x[7] = x[2] * x[4] * (x[4]**2 - 3.0 * x[6]**2) - x[0] * x[6] * (x[6]**2 - 3.0 * x[4]**2) + x[3] * x[5] * (x[5]**2 - 3.0 * x[7]**2) - x[1] * x[7] * (x[7]**2 - 3.0 * x[5]**2) - 9.48

            return f_x


    elif functionNumber == 23:  # STYLBINSKI-TANG function
        @modificator
        def f(x):
            f_x = zeros((1, 1))

            f_x[0] = sum(x**4 - 16 * x**2 + 5 * x) / 2

            fmin = - 39.1661657037714 * x.size

            f_x = f_x - fmin

            return f_x


    elif functionNumber == 24:  # RASTRIGIN function
        @modificator
        def f(x):
            f_x = zeros((1, 1))

            f_x[0] = 10 * x.size + sum(x**2 - 10 * cos(2 * pi * x))

            return f_x


    elif functionNumber == 25: # GOLDTSEIN-PRICE function
        @modificator
        def f(x):
            f_x = zeros((1, 1))

            f_x[0] = (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2))
            f_x[0] = f_x[0] * (30 + (2 * x[0] - 2 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
            f_x[0] = f_x[0] - 3

            return f_x


    elif functionNumber == 26: # BRANIN function
        @modificator
        def f(x):
            f_x = zeros((1, 1))

            a = 1
            b = 5.1 / (4 * pi**2)
            c = 5 / pi
            r = 6
            s = 10
            t = 1 / (8 * pi)

            f_x[0] = a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * cos(x[0]) + s

            fmin = 0.39788735772973816

            f_x = f_x - fmin

            return f_x


    elif functionNumber == 27:  # ACKLEY function
        @modificator
        def f(x):
            f_x = zeros((1, 1))

            a = 20
            b = 0.2
            c = 2 * pi
            d = x.size

            f_x[0] = -a * exp(-b * sqrt( 1 / d * sum(x**2))) - exp(1 / d + sum(cos(c * x))) + a + exp(1)

            return f_x


    elif functionNumber == 28:  # HARTMAN3 function
        @modificator
        def f(x):
            a = array([1, 1.2, 3, 3.2])

            A = array([[3, 10, 30],
                          [0.1, 10, 35],
                          [3, 10, 30],
                          [0.1, 10, 35]])

            P = 1e-4 * array([[3689, 1170, 2673],
                                 [4699, 4387, 7470],
                                 [1091, 8732, 5547],
                                 [381, 5743, 8828]])

            f_x = zeros((1, 1))

            for i in range(4):
                t = 0

                for j in range(3):
                    t = t + A[i, j] * (x[j] - P[i, j])**2

                f_x[0] = f_x[0] + a[i] * exp(-t)

            f_x = - f_x

            fmin = - 3.8627797869493365

            f_x = f_x - fmin

            return f_x


    elif functionNumber == 29:  # HARTMAN6 function
        @modificator
        def f(x):
            a = array([1, 1.2, 3, 3.2])

            A = array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])

            P = 1e-4 * array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

            f_x = zeros((1, 1))

            for i in range(4):
                t = 0

                for j in range(6):
                    t = t + A[i, j] * (x[j] - P[i, j])**2

                f_x[0] = f_x[0] + a[i] * exp(-t)

            f_x = - f_x

            fmin = - 3.322368011391339

            f_x = f_x - fmin

            return f_x

    elif functionNumber == 30: #SHEKEL function
        @modificator
        def f(x):
            b = 1 / 10 * array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])

            C = array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                          [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                          [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                          [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])

            f_x = zeros((1, 1))

            for i in range(m):
                t = b[i]

                for j in range(4):
                    t = t + (x[j] - C[j, i])**2

                f_x[0] = f_x[0] + 1 / t

            f_x = - f_x

            if m == 5:
                fmin = - 10.153195850979039
            elif m == 7:
                fmin = - 10.402818836930305
            elif m == 10:
                fmin = - 10.536283726219603

            f_x = f_x - fmin

            return f_x


    elif functionNumber == 31: # WOOD function
        @modificator
        def f(x):
            f_x = zeros((6, 1))

            f_x[0] = 10 * (x[1] - x[0]**2)
            f_x[1] = 1 - x[0]
            f_x[2] = sqrt(90) * (x[3] - x[2]**2)
            f_x[3] = 1 - x[2]
            f_x[4] = sqrt(10) * (x[1] + x[3] - 2)
            f_x[5] = sqrt(10) * (x[1] - x[3])

            return f_x

    else:
        raise ValueError("Function number {} not defined.".format(functionNumber))

    return f
