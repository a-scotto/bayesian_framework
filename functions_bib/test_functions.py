# -*- coding: utf-8 -*-

##########################################################################
#                               BENCHMARK	                                   #
##########################################################################
# \brief                fonctions academiques
#
# \author Mohamed Amine Bouhlel
# Bouhlel Mohamed Amine <mohamed.bouhlel@snecma.fr>
#                       <mohamed.amine.bouhlel@gmail.com>
import os
import time
import math
import numpy as np

"""
FUNCTION    : DIMENSION

Ackley      : 2
Branin      : 2
Easom       : 2
Himmelblau  : 2

Carre       : n
DixonPrice : n
Griewank    : n
Michalewicz : n
Rana        : n
Rastrigin   : n
Rosenbrock  : n
Styblinski  : n
"""


def Himmelblau(X):

    return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2


def Easom(X):

    return -np.cos(X[0]) * np.cos(X[1]) * np.exp(-((X[0] - np.pi)**2 + (X[1] - np.pi)**2))


def Rastrigin(X):
    A = 10
    return A * X.size + sum(X**2 - A * np.cos(2 * np.pi * X))


def Styblinski(X):

    return sum(X**4 - 16 * X**2 + 5 * X) / 2


def Michalewicz(X, m=10):
    """ PARAMETERS : 

            X : row vector (array) of length d
            m : float, recommended value m = 10
    """

    i = np.arange(0, X.size) + 1
    return -sum(np.sin(X) * np.sin(i * X**2 / np.pi)**(2 * m))


def Branin(X):
    """ PARAMETERS : 

            X : row vector (array) of length 2
    """

    # Recommended values for the Branin funtion parmaters
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    return a * (X[1] - b * X[0]**2 + c * X[0] - r)**2 + s * (1 - t) * np.cos(X[0]) + s


def Mopta(x, path_exec):

    n, dim = x.shape
    y = np.zeros((n, 69))
    # y = np.zeros((n,54))

    for i in range(n):
        x_input = open("input.txt", "w")
        for j in range(dim):
            x_input.write(str(x[i, j]) + "\n")

        x_input.close()

        os.system(path_exec)
        t1 = time.time()
        while(True):
            t2 = time.time()
            if t2 - t1 > 4:
                break

        y_output = open("output.txt", "r")
        lines = y_output.readlines()
       # incr = 0#a virer
        for j in range(69):
            # if j not in [2,17,18,23,29,37,39,41,47,49,51,53,55,64,65]:#a
            # virer cette ligne

            line = lines[j].split()
            y[i, j] = float(line[0])
            # else:
            # incr+=1
        y_output.close()

    return y, {}


def Mopta_12d(x, path_exec):

    n, dim = x.shape

    y = np.zeros((n, 69))
    refsol = np.zeros((n, 124))
    refsol += np.array([[0.424901536845442, 0.000000000000000, -0.000000000000000,
                         0.000000000000000, 0.000000000000000, 0.070368163337397,
                         0.191758133431455, 0.659581587650616, 0.312511963893922,
                         1.000000000000000, 0.000000000000000, -0.000000000000000,
                         0.000000000000000, 0.000000000000000, -0.000000000000000,
                         0.500523655520237, 0.000000000000000, 0.000000000000000,
                         0.000000000000000, -0.000000000000000, -0.000000000000000,
                         0.007323697294187, -0.000000000000000, 0.000000000000000,
                         0.000000000000000, 0.691424942460969, 0.549284506769106,
                         -0.000000000000000, -0.000000000000000, 0.634896580136233,
                         0.246440520422731, 0.312017956345506, 0.000000000000000,
                         1.000000000000000, 1.000000000000000, 0.317952882489170,
                         0.855170249994022, 0.658962493142454, 0.567339401149320,
                         0.191865333482370, 0.906569943802102, 0.612762717451940,
                         0.033974026079705, 0.250858210690466, -0.000000000000000,
                         0.000000000000000, 0.397548021112299, 0.050453268214836,
                         0.144718492777280, 0.065661350680142, -0.000000000000000,
                         -0.000000000000000, 0.401813779445987, -0.000000000000000,
                         0.000000000000000, 0.181619715253390, 0.000000000000000,
                         -0.000000000000000, 0.114195768540203, 0.393419805218654,
                         0.047400618876390, 0.000000000000000, 0.877814656956643,
                         0.767294038418826, 0.000000000000000, 0.853786893241821,
                         0.603638877543724, 0.330387862928772, 0.000000000000000,
                         0.204560994583241, 0.221383426733140, 0.000000000000000,
                         0.000000000000000, 0.094398339150782, -0.000000000000000,
                         0.308945500999398, -0.000000000000000, 0.320954495292004,
                         0.735573944494521, 0.000000000000000, 0.510228205500305,
                         0.558204538496431, 0.527207739609047, 0.827802688106254,
                         -0.000000000000000, 0.665687979989158, 0.724380930097933,
                         0.287131241910361, 0.414606427865539, -0.000000000000000,
                         0.419521821597494, 0.388002834344305, 0.029815463457816,
                         0.631619810505481, -0.000000000000000, 0.000000000000000,
                         0.192333290055701, 0.000000000000000, 0.692543616440458,
                         0.669383104588378, -0.000000000000000, 1.000000000000000,
                         0.729141260482755, 0.161236677500071, 0.561446507760961,
                         0.487479618244102, 0.728572274406474, 0.240635584133437,
                         -0.000000000000000, 1.000000000000000, 0.383816619144691,
                         1.000000000000000, 1.000000000000000, 0.000000000000000,
                         1.000000000000000, 0.152841300847909, 0.158379914361422,
                         0.210185493857930, 0.889704350522390, 0.791266802460270,
                         -0.000000000000000, 0.931445345989215, -0.000000000000000,
                         -0.000000000000000]])
    # varindices = [0,1,5,10,22,24,27,33,35,38,40,41,44,45,47,48,50,55,59,60,63,65,71,77,78,79,81,84,87,90,92,93,94,98,100,105,107,110,111,113,118,120,122,123]
    varindices = [1, 41, 44, 45, 48, 50, 59, 65, 71, 77, 81, 94]
    refsol[:, varindices] = x

    for i in range(n):
        x_input = open("input.txt", "w")
        for j in range(124):
            x_input.write(str(refsol[i, j]) + "\n")
        x_input.close()

        os.system(path_exec)

        t1 = time.time()
        while(True):
            t2 = time.time()
            if t2 - t1 > 4:
                break

        y_output = open("output.txt", "r")
        lines = y_output.readlines()
       # incr = 0#a virer
        for j in range(69):
            # if j not in [2,17,18,23,29,37,39,41,47,49,51,53,55,64,65]:#a
            # virer cette ligne
            line = lines[j].split()
            y[i, j] = float(line[0])
            # else:
            # incr+=1
        y_output.close()

    return y, {}


def Mopta_30d(x, path_exec):

    n, dim = x.shape

    y = np.zeros((n, 69))
    refsol = np.zeros((n, 124))
    refsol += np.array([[0.424901536845442, 0.000000000000000, -0.000000000000000, 0.000000000000000, 0.000000000000000, 0.070368163337397, 0.191758133431455, 0.659581587650616, 0.312511963893922, 1.000000000000000, 0.000000000000000, -0.000000000000000, 0.000000000000000, 0.000000000000000, -0.000000000000000, 0.500523655520237, 0.000000000000000, 0.000000000000000, 0.000000000000000, -0.000000000000000, -0.000000000000000, 0.007323697294187, -0.000000000000000, 0.000000000000000, 0.000000000000000, 0.691424942460969, 0.549284506769106, -0.000000000000000, -0.000000000000000, 0.634896580136233, 0.246440520422731, 0.312017956345506, 0.000000000000000, 1.000000000000000, 1.000000000000000, 0.317952882489170, 0.855170249994022, 0.658962493142454, 0.567339401149320, 0.191865333482370, 0.906569943802102, 0.612762717451940, 0.033974026079705, 0.250858210690466, -0.000000000000000, 0.000000000000000, 0.397548021112299, 0.050453268214836, 0.144718492777280, 0.065661350680142, -0.000000000000000, -0.000000000000000, 0.401813779445987, -0.000000000000000, 0.000000000000000, 0.181619715253390, 0.000000000000000, -0.000000000000000, 0.114195768540203, 0.393419805218654, 0.047400618876390, 0.000000000000000,
                         0.877814656956643, 0.767294038418826, 0.000000000000000, 0.853786893241821, 0.603638877543724, 0.330387862928772, 0.000000000000000, 0.204560994583241, 0.221383426733140, 0.000000000000000, 0.000000000000000, 0.094398339150782, -0.000000000000000, 0.308945500999398, -0.000000000000000, 0.320954495292004, 0.735573944494521, 0.000000000000000, 0.510228205500305, 0.558204538496431, 0.527207739609047, 0.827802688106254, -0.000000000000000, 0.665687979989158, 0.724380930097933, 0.287131241910361, 0.414606427865539, -0.000000000000000, 0.419521821597494, 0.388002834344305, 0.029815463457816, 0.631619810505481, -0.000000000000000, 0.000000000000000, 0.192333290055701, 0.000000000000000, 0.692543616440458, 0.669383104588378, -0.000000000000000, 1.000000000000000, 0.729141260482755, 0.161236677500071, 0.561446507760961, 0.487479618244102, 0.728572274406474, 0.240635584133437, -0.000000000000000, 1.000000000000000, 0.383816619144691, 1.000000000000000, 1.000000000000000, 0.000000000000000, 1.000000000000000, 0.152841300847909, 0.158379914361422, 0.210185493857930, 0.889704350522390, 0.791266802460270, -0.000000000000000, 0.931445345989215, -0.000000000000000, -0.000000000000000]])
    # varindices = [0,1,5,10,22,24,27,33,35,38,40,41,44,45,47,48,50,55,59,60,63,65,71,77,78,79,81,84,87,90,92,93,94,98,100,105,107,110,111,113,118,120,122,123]
    varindices = [1, 2, 5, 12, 13, 14, 21, 23, 28, 32, 33, 35, 41, 44,
                  45, 48, 50, 53, 59, 65, 68, 71, 77, 81, 84, 85, 94, 95, 100, 102]
    refsol[:, varindices] = x

    for i in range(n):

        x_input = open("input.txt", "w")
        for j in range(124):
            x_input.write(str(refsol[i, j]) + "\n")

        x_input.close()

        os.system(path_exec)

        t1 = time.time()
        while(True):
            t2 = time.time()
            if t2 - t1 > 4:
                break

        y_output = open("output.txt", "r")
        lines = y_output.readlines()
       # incr = 0#a virer
        for j in range(69):
            # if j not in [2,17,18,23,29,37,39,41,47,49,51,53,55,64,65]:#a
            # virer cette ligne

            line = lines[j].split()
            y[i, j] = float(line[0])
            # else:
            # incr+=1
        y_output.close()

    return y, {}


def Mopta_50d(x, path_exec):

    n, dim = x.shape

    y = np.zeros((n, 69))
    refsol = np.zeros((n, 124))
    refsol += np.array([[0.424901536845442, 0.000000000000000, -0.000000000000000, 0.000000000000000, 0.000000000000000, 0.070368163337397, 0.191758133431455, 0.659581587650616, 0.312511963893922, 1.000000000000000, 0.000000000000000, -0.000000000000000, 0.000000000000000, 0.000000000000000, -0.000000000000000, 0.500523655520237, 0.000000000000000, 0.000000000000000, 0.000000000000000, -0.000000000000000, -0.000000000000000, 0.007323697294187, -0.000000000000000, 0.000000000000000, 0.000000000000000, 0.691424942460969, 0.549284506769106, -0.000000000000000, -0.000000000000000, 0.634896580136233, 0.246440520422731, 0.312017956345506, 0.000000000000000, 1.000000000000000, 1.000000000000000, 0.317952882489170, 0.855170249994022, 0.658962493142454, 0.567339401149320, 0.191865333482370, 0.906569943802102, 0.612762717451940, 0.033974026079705, 0.250858210690466, -0.000000000000000, 0.000000000000000, 0.397548021112299, 0.050453268214836, 0.144718492777280, 0.065661350680142, -0.000000000000000, -0.000000000000000, 0.401813779445987, -0.000000000000000, 0.000000000000000, 0.181619715253390, 0.000000000000000, -0.000000000000000, 0.114195768540203, 0.393419805218654, 0.047400618876390, 0.000000000000000,
                         0.877814656956643, 0.767294038418826, 0.000000000000000, 0.853786893241821, 0.603638877543724, 0.330387862928772, 0.000000000000000, 0.204560994583241, 0.221383426733140, 0.000000000000000, 0.000000000000000, 0.094398339150782, -0.000000000000000, 0.308945500999398, -0.000000000000000, 0.320954495292004, 0.735573944494521, 0.000000000000000, 0.510228205500305, 0.558204538496431, 0.527207739609047, 0.827802688106254, -0.000000000000000, 0.665687979989158, 0.724380930097933, 0.287131241910361, 0.414606427865539, -0.000000000000000, 0.419521821597494, 0.388002834344305, 0.029815463457816, 0.631619810505481, -0.000000000000000, 0.000000000000000, 0.192333290055701, 0.000000000000000, 0.692543616440458, 0.669383104588378, -0.000000000000000, 1.000000000000000, 0.729141260482755, 0.161236677500071, 0.561446507760961, 0.487479618244102, 0.728572274406474, 0.240635584133437, -0.000000000000000, 1.000000000000000, 0.383816619144691, 1.000000000000000, 1.000000000000000, 0.000000000000000, 1.000000000000000, 0.152841300847909, 0.158379914361422, 0.210185493857930, 0.889704350522390, 0.791266802460270, -0.000000000000000, 0.931445345989215, -0.000000000000000, -0.000000000000000]])
    # varindices = [0,1,5,10,22,24,27,33,35,38,40,41,44,45,47,48,50,55,59,60,63,65,71,77,78,79,81,84,87,90,92,93,94,98,100,105,107,110,111,113,118,120,122,123]
    varindices = [1, 2, 5, 10, 12, 13, 14, 18, 20, 21, 23, 28, 29, 30, 32, 33, 35, 39, 41, 44, 45, 48, 50, 52,
                  53, 55, 56, 59, 61, 62, 64, 65, 68, 70, 71, 75, 76, 77, 80, 81, 84, 85, 91, 92, 94, 95, 100, 102, 120, 123]

    refsol[:, varindices] = x

    for i in range(n):

        x_input = open("input.txt", "w")
        for j in range(124):
            x_input.write(str(refsol[i, j]) + "\n")

        x_input.close()

        os.system(path_exec)

        t1 = time.time()
        while(True):
            t2 = time.time()
            if t2 - t1 > 4:
                break

        y_output = open("output.txt", "r")
        lines = y_output.readlines()
       # incr = 0#a virer
        for j in range(69):
            # if j not in [2,17,18,23,29,37,39,41,47,49,51,53,55,64,65]:#a
            # virer cette ligne

            line = lines[j].split()
            y[i, j] = float(line[0])
            # else:
            # incr+=1
        y_output.close()

    return y, {}


def Rana(X):
    """
    x in [-5.12,5.12]
    """
    y = 0

    for i in range(X.size - 1):
        y += X[i] * np.cos(np.sqrt(np.abs(X[i + 1] + X[i] + 1))) * np.sin(np.sqrt(np.abs(X[i + 1] - X[i] + 1))) + (
            1 + X[i + 1]) * np.sin(np.sqrt(np.abs(X[i + 1] + X[i] + 1))) * np.cos(np.sqrt(np.abs(X[i + 1] - X[i] + 1)))

    return y


def Rosenbrock(X):

    y = 0

    for i in range(X.size - 1):
        y += 100 * (X[i + 1] - X[i]**2)**2 + (X[i] - 1)**2

    return y


def Carre(X):

    return sum(X**2)


def Ackley(X):

    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (X[0]**2 + X[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * X[0]) + np.cos(2 * np.pi * X[1]))) + np.exp(1) + 20


def DixonPrice(X):

    y = (X[0] - 1)**2

    for i in range(X.size - 2):
        y += (i + 2) * (X[i + 1] - X[i]**2)**2 + (X[i] - 1)**2

    return y


def Griewank(X):
    """
    Fonction Griewank:
    Entrees x (n_evals,dimension)
    """
    i = np.arange(0, X.size) + 1
    f = 4000

    return np.sum(X**2 / f) - np.prod(np.cos(X / i)) + 1


def tird(x):
    """
    Fonction Tird:
    Entrees x (n_evals,dimension)
    """

    n, dim = x.shape

    s1 = np.zeros((n, 1))
    s2 = np.zeros((n, 1))
    s1 = array2d(np.sum((x - 1)**2, 1)).T
    for i in range(1, dim):
        s2 = s2 + array2d(x[:, i - 1] * x[:, i]).T

    return array2d(s1 - s2)


def G07(x):
    """
    Fonction G07 with y[:,traints:
    Entrees x (n_evals,dimension)
    """

    n, dim = x.shape
    if dim != 10:
        print("dimension must be equal 10")
        raise

    y = np.zeros((n, 9))
    y[:, 0] = x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1] - 14 * x[:, 0] - 16 * x[:, 1] + \
        (x[:, 2] - 10)**2 + 4 * (x[:, 3] - 5)**2 + (x[:, 4] - 3)**2 + 2 * (x[:, 5] - 1)**2 + 5 * \
        x[:, 6]**2 + 7 * (x[:, 7] - 11)**2 + 2 * \
        (x[:, 8] - 10)**2 + (x[:, 9] - 7)**2 + 45

    y[:, 1] = 4 * x[:, 0] + 5 * x[:, 1] - 3 * x[:, 6] + 9 * x[:, 7] - 105
    y[:, 2] = 10 * x[:, 0] - 8 * x[:, 1] - 17 * x[:, 6] + 2 * x[:, 7]
    y[:, 3] = -8 * x[:, 0] + 2 * x[:, 1] + 5 * x[:, 8] - 2 * x[:, 9] - 12
    y[:, 4] = 3 * (x[:, 0] - 2)**2 + 4 * (x[:, 1] - 3)**2 + \
        2 * x[:, 2]**2 - 7 * x[:, 3] - 120
    y[:, 5] = 5 * x[:, 0]**2 + 8 * x[:, 1] + \
        (x[:, 2] - 6)**2 - 2 * x[:, 3] - 40
    y[:, 7] = 0.5 * (x[:, 0] - 8)**2 + 2 * (x[:, 1] - 4)**2 + \
        3 * x[:, 4]**2 - x[:, 5] - 30
    y[:, 6] = x[:, 0]**2 + 2 * (x[:, 1] - 2)**2 - 2 * \
        x[:, 0] * x[:, 1] + 14 * x[:, 4] - 6 * x[:, 5]
    y[:, 8] = -3 * x[:, 0] + 6 * x[:, 1] + 12 * (x[:, 8] - 8)**2 - 7 * x[:, 9]

    return y, {}


def G07MOD(x):
    """
    Modification fonction G07 with y[:,traints:
    Entrees x (n_evals,dimension)
    """

    n, dim = x.shape
    if dim != 10:
        print("dimension must be equal 10")
        raise

    y = np.zeros((n, 9))
    y[:, 0] = x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1] - 14 * x[:, 0] - 16 * x[:, 1] + \
        (x[:, 2] - 10)**2 + 4 * (x[:, 3] - 5)**2 + (x[:, 4] - 3)**2 + 2 * (x[:, 5] - 1)**2 + 5 * \
        x[:, 6]**2 + 7 * (x[:, 7] - 11)**2 + 2 * \
        (x[:, 8] - 10)**2 + (x[:, 9] - 7)**2 + 45

    y[:, 1] = (4 * x[:, 0] + 5 * x[:, 1] - 3 *
               x[:, 6] + 9 * x[:, 7] - 105) / 105.
    y[:, 2] = (10 * x[:, 0] - 8 * x[:, 1] - 17 * x[:, 6] + 2 * x[:, 7]) / 370.
    y[:, 3] = (-8 * x[:, 0] + 2 * x[:, 1] + 5 *
               x[:, 8] - 2 * x[:, 9] - 12) / 158.
    y[:, 4] = (3 * (x[:, 0] - 2)**2 + 4 * (x[:, 1] - 3) **
               2 + 2 * x[:, 2]**2 - 7 * x[:, 3] - 120) / 1258.
    y[:, 5] = (5 * x[:, 0]**2 + 8 * x[:, 1] +
               (x[:, 2] - 6)**2 - 2 * x[:, 3] - 40) / 816.
    y[:, 6] = (0.5 * (x[:, 0] - 8)**2 + 2 * (x[:, 1] - 4)
               ** 2 + 3 * x[:, 4]**2 - x[:, 5] - 30) / 834.
    y[:, 7] = (x[:, 0]**2 + 2 * (x[:, 1] - 2)**2 - 2 * x[:, 0]
               * x[:, 1] + 14 * x[:, 4] - 6 * x[:, 5]) / 788.
    y[:, 8] = (-3 * x[:, 0] + 6 * x[:, 1] + 12 *
               (x[:, 8] - 8)**2 - 7 * x[:, 9]) / 4048.

    return y, {}


def mystery(x):

    n, dim = x.shape
    if dim != 2:
        print("dimension must be equal 2")
        raise

    y = np.zeros((n, 2))
    y[:, 0] = 2 + 0.01 * (x[:, 1] - x[:, 0]**2)**2 + (1 - x[:, 0])**2 + 2 * \
        (2 - x[:, 1])**2 + 7 * np.sin(0.5 * x[:, 0]) * \
        np.sin(0.7 * x[:, 0] * x[:, 1])

    y[:, 1] = - np.sin(x[:, 0] - x[:, 1] - np.pi / 8.)

    return y


def function2(x):
    # x_i in [0,1]^2 for i = 1,2

    n, dim = x.shape
    if dim != 2:
        print("dimension must be equal 2")
        raise

    y = np.zeros((n, 4))
    y[:, 0] = - (x[:, 0] - 1)**2 - (x[:, 1] - 0.5)**2
    y[:, 1] = (x[:, 0] - 3)**2 + (x[:, 1] + 2)**2 * np.exp(-x[:, 1]**7) - 12
    y[:, 2] = 10 * x[:, 0] + x[:, 1] - 7
    y[:, 3] = (x[:, 0] - 0.5)**2 + (x[:, 1] - 0.5)**2 - 0.2

    return y


def g06(x):
    n, dim = x.shape
    if dim != 2:
        print("dimension must be equal 2")
        raise

    y = np.zeros((n, 3))
    y[:, 0] = (x[:, 0] - 10)**3 + (x[:, 1] - 20)**3
    y[:, 1] = (-(x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100) / 100
    y[:, 2] = ((x[:, 0] - 6)**2 - (x[:, 1] - 5)**2 + 82.81) / 82.81

    return y, {}


def newBranin(x):
    # xi in [-5,10]*[0,15]

    n, dim = x.shape
    if dim != 2:
        print("dimension must be equal 2")
        raise

    y = np.zeros((n, 2))
    y[:, 0] = - (x[:, 0] - 10)**2 - (x[:, 1] - 15)**2
    y[:, 1] = (x[:, 1] - 5.1 / (4 * np.pi**2) * x[:, 0]**2 + 5. / np.pi * x[:, 0] - 6)**2 + 10 * \
        (1 - 1. / (8 * np.pi)) * np.cos(x[:, 0]) - 5

    return y


def g2(x):
    """
    binf = [0]
    bsup = [10]
    for i in range(dim):
        binf.append(0)
        bsup.append(10)

    """

    n, dim = x.shape
    y = np.zeros((n, 3))
    ii = np.zeros((1, dim))
    for i in range(dim):
        ii[0, i] = i + 1

    y[:, 0] = -np.abs((np.sum(np.cos(x)**4, 1) - 2. *
                       np.prod(np.cos(x)**2, 1)) / np.sqrt(np.sum(ii * x**2, 1)))
    for i in range(n):
        z = -np.prod(x[i, :]) + 0.75
        if z >= 0:
            y[i, 1] = np.log(1 + z)
        else:
            y[i, 1] = -np.log(1 - z)

    y[:, 2] = (np.sum(x, 1) - 7.5 * dim) / (2.5 * dim)

    return y, {}


def g1(x):

    n, dim = x.shape
    y = np.zeros((n, 10))

    y[:, 0] = 5 * np.sum(x[:, 0:4], 1) - 5 * \
        np.sum(x[:, 0:4]**2, 1) - np.sum(x[:, 4:], 1)
    y[:, 1] = 2 * x[:, 0] + 2 * x[:, 1] + x[:, 9] + x[:, 10] - 10
    y[:, 2] = 2 * x[:, 0] + 2 * x[:, 2] + x[:, 9] + x[:, 11] - 10
    y[:, 3] = 2 * x[:, 1] + 2 * x[:, 2] + x[:, 10] + x[:, 11] - 10
    y[:, 4] = -8 * x[:, 0] + x[:, 9]
    y[:, 5] = -8 * x[:, 1] + x[:, 10]
    y[:, 6] = -8 * x[:, 2] + x[:, 11]
    y[:, 7] = -2 * x[:, 3] - x[:, 4] + x[:, 9]
    y[:, 8] = -2 * x[:, 5] - x[:, 6] + x[:, 10]
    y[:, 9] = -2 * x[:, 7] - x[:, 8] + x[:, 11]
    return y


def WWF(X):
    """
    Wing Weight Function

    binf=[150]
    binf.append(220)
    binf.append(6)
    binf.append(-10)
    binf.append(16)
    binf.append(0.5)
    binf.append(0.08)
    binf.append(2.5)
    binf.append(1700)
    binf.append(0.025)

    bsup = [200]
    bsup.append(300)
    bsup.append(10)
    bsup.append(10)
    bsup.append(45)
    bsup.append(1)
    bsup.append(0.18)
    bsup.append(6)
    bsup.append(2500)
    bsup.append(0.08)
    """

    n, dim = X.shape
    if dim != 10:
        print("dimension must be equal 10")
        raise

   # if (X[:,0] < 150) or (X[:,0] > 200) or (X[:,1] < 220) or (X[:,1] > 300) or (X[:,2] < 6) or (X[:,2] > 10) or (X[:,3] < -10) or (X[:,3] > 10) or (X[:,4] < 16) or (X[:,4] > 45) or (X[:,5] < 0.5) or (X[:,5] > 1) or (X[:,6] < 0.08) or (X[:,6] > 0.18) or (X[:,7] < 2.5) or (X[:,7] > 6) or (X[:,8] < 1700) or (X[:,8] > 2500) or (X[:,9] < 0.025) or (X[:,9] > 0.08):
   #     print "Bounds of variables are not respected"
   #     raise

    return 0.036 * X[:, 0]**0.758 * X[:, 1]**0.0035 * (X[:, 2] / np.cos(np.radians(X[:, 3]))**2)**0.6 * X[:, 4]**0.006 * X[:, 5]**0.04 * (100 * X[:, 6] / np.cos(np.radians(X[:, 3])))**(-0.3) * (X[:, 7] * X[:, 8])**0.49 + X[:, 0] * X[:, 9]


def WB4(x):
    """
    Welded Beam
    Best solution : f = 1.728226
    x* = [0.20564426101885,3.47257874213172,9.03662391018928,0.20572963979791]

    binf=[0.125]
    binf.append(0.1)
    binf.append(0.1)
    binf.append(0.1)

    bsup = [10]
    bsup.append(10)
    bsup.append(10)
    bsup.append(10)
    """

    n, dim = x.shape
    if dim != 4:
        print("dimension must be equal 4")
        raise

    # Data
    P = 6000.
    L = 14.
    E = 30 * 1e6
    G = 12 * 1e6
    tmax = 13600.
    smax = 30000.
    xmax = 10.
    dmax = 0.25
    M = P * (L + x[:, 1] / 2.)
    R = np.sqrt(0.25 * (x[:, 1]**2 + (x[:, 0] + x[:, 2])**2))
    J = np.sqrt(2) * x[:, 0] * x[:, 1] * (x[:, 1] **
                                          2 / 12. + 0.25 * (x[:, 0] + x[:, 2])**2)
    Pc = 4.013 * E / (6 * L**2) * x[:, 2] * x[:, 3]**3 * \
        (1 - 0.25 * x[:, 2] * np.sqrt(E / G) / L)
    t1 = P / (np.sqrt(2) * x[:, 0] * x[:, 1])
    t2 = M * R / J
    t = np.sqrt(t1**2 + t1 * t2 * x[:, 1] / R + t2**2)
    s = 6 * P * L / (x[:, 3] * x[:, 2]**2)
    d = 4 * P * L**3 / (E * x[:, 3] * x[:, 2]**3)

    y = np.zeros((n, 7))
    y[:, 0] = 1.10471 * x[:, 0]**2 * x[:, 1] + \
        0.04811 * x[:, 2] * x[:, 3] * (14 + x[:, 1])
    y[:, 1] = (t - tmax) / tmax
    y[:, 2] = (s - smax) / smax
    y[:, 3] = (x[:, 0] - x[:, 3]) / xmax
    y[:, 4] = (0.10471 * x[:, 0]**2 + 0.04811 * x[:, 2]
               * x[:, 3] * (14 + x[:, 1]) - 5) / 5.
    y[:, 5] = (d - dmax) / dmax
    y[:, 6] = (P - Pc) / P

    return y, {}


def PVD4(x):
    """
    Pressure Vessel Design
    binf = [0]
    binf.append(0)
    binf.append(0)
    binf.append(0)

    bsup = [1]
    bsup.append(1)
    bsup.append(50)
    bsup.append(240)
    """

    n, dim = x.shape
    if dim != 4:
        print("dimension must be equal 4")
        raise

    y = np.zeros((n, 4))
    y[:, 0] = 0.6224 * x[:, 0] * x[:, 2] * x[:, 3] + 1.7781 * x[:, 1] * \
        x[:, 2]**2 + 3.1661 * x[:, 0]**2 * \
        x[:, 3] + 19.84 * x[:, 0]**2 * x[:, 2]
    y[:, 1] = -x[:, 0] + 0.0193 * x[:, 2]
    y[:, 2] = -x[:, 1] + 0.00954 * x[:, 2]
    for i in range(n):
        z = -np.pi * x[i, 2]**2 * x[i, 3] - \
            4. / 3 * np.pi * x[i, 2]**3 + 1296000
        if z >= 0:
            y[i, 3] = np.log(1 + z)
        else:
            y[i, 3] = -np.log(1 - z)

    return y, {}


def SR7(x):
    """
    Spead reducer
    binf = [2.6]
    binf.append(0.7)
    binf.append(17.)
    binf.append(7.3)
    binf.append(7.3)
    binf.append(2.9)
    binf.append(5.)

    bsup = [3.6]
    bsup.append(0.8)
    bsup.append(28)
    bsup.append(8.3)
    bsup.append(8.3)
    bsup.append(3.9)
    bsup.append(5.5)
    """

    n, dim = x.shape
    if dim != 7:
        print("dimension must be equal 7")
        raise

    y = np.zeros((n, 12))
    A = 3.3333 * x[:, 2]**2 + 14.9334 * x[:, 2] - 43.0934
    B = x[:, 5]**2 + x[:, 6]**2
    C = x[:, 5]**3 + x[:, 6]**3
    D = x[:, 3] * x[:, 5]**2 + x[:, 4] * x[:, 6]**2
    A1 = ((745 * x[:, 3] / (x[:, 1] * x[:, 2]))**2 + (16.91 * 1e6))**0.5
    B1 = 0.1 * x[:, 5]**3
    A2 = ((745 * x[:, 4] / (x[:, 1] * x[:, 2]))**2 + (157.5 * 1e6))**0.5
    B2 = 0.1 * x[:, 6]**3
    y[:, 0] = 0.7854 * x[:, 0] * x[:, 1]**2 * A - \
        1.508 * x[:, 0] * B + 7.477 * C + 0.7854 * D
    y[:, 1] = (27 - x[:, 0] * x[:, 1]**2 * x[:, 2]) / 27.
    y[:, 2] = (397.5 - x[:, 0] * x[:, 1]**2 * x[:, 2]**2) / 397.5
    y[:, 3] = (1.93 - (x[:, 1] * x[:, 5]**4 * x[:, 2]) / x[:, 3]**3) / 1.93
    y[:, 4] = (1.93 - (x[:, 1] * x[:, 6]**4 * x[:, 2]) / x[:, 4]**3) / 1.93
    y[:, 5] = ((A1 / B1) - 1100) / 1100.
    y[:, 6] = ((A2 / B2) - 850) / 850.
    y[:, 7] = (x[:, 1] * x[:, 2] - 40) / 40.
    y[:, 8] = (5 - (x[:, 0] / x[:, 1])) / 5.
    y[:, 9] = ((x[:, 0] / x[:, 1]) - 12) / 12.
    y[:, 10] = (1.9 + 1.5 * x[:, 5] - x[:, 3]) / 1.9
    y[:, 11] = (1.9 + 1.1 * x[:, 6] - x[:, 4]) / 1.9

    return y, {}


def GTCD(x):
    """
    Gas Transmission Design
    binf = [20]
    binf.append(1)
    binf.append(20)
    binf.append(0.1)

    bsup = [50]
    bsup.append(10)
    bsup.append(50)
    bsup.append(60)
    """

    n, dim = x.shape
    if dim != 4:
        print("dimension must be equal 4")
        raise

    y = np.zeros((n, 2))
    y[:, 0] = (8.61 * 1e5) * x[:, 0]**(0.5) * x[:, 1] * x[:, 2]**(-2.0 / 3) * x[:, 3]**(-0.5) + \
        (3.69 * 1e4) * x[:, 2] + (7.72 * 1e8) * x[:, 0]**- \
        1 * x[:, 1]**0.219 - (765.43 * 1e6) * x[:, 0]**-1
    y[:, 1] = x[:, 3] * x[:, 1]**-2 + x[:, 1]**-2 - 1

    return y, {}


def G3MOD(x):
    """
    G3MOD
    binf = [0]
    bsup = [1]
    for i in range(dim):
        binf.append(0)
        bsup.append(1)
    """

    n, dim = x.shape

    y = np.zeros((n, 2))
    for i in range(n):
        z = np.sqrt(dim)**dim * np.prod(x[i, :])
        if z >= 0:
            y[i, 0] = - np.log(1 + z)
        else:
            y[i, 0] = np.log(1 - z)
    y[:, 1] = np.sum(x**2, 1) - 1

    return y, {}


def G4(x):
    """
    G4
    binf = [78]
    binf.append(33)
    binf.append(27)
    binf.append(27)
    binf.append(27)

    bsup = [102]
    bsup.append(45)
    bsup.append(45)
    bsup.append(45)
    bsup.append(45)
    """

    n, dim = x.shape
    if dim != 5:
        print("dimension must be equal 5")
        raise

    y = np.zeros((n, 7))
    u = 85.334407 + 0.0056858 * \
        x[:, 1] * x[:, 4] + 0.0006262 * x[:, 0] * \
        x[:, 3] - 0.0022053 * x[:, 2] * x[:, 4]
    v = 80.51249 + 0.0071317 * x[:, 1] * x[:, 4] + \
        0.0029955 * x[:, 0] * x[:, 1] + 0.0021813 * x[:, 2]**2
    w = 9.300961 + 0.0047026 * x[:, 2] * x[:, 4] + 0.0012547 * \
        x[:, 0] * x[:, 2] + 0.0019085 * x[:, 2] * x[:, 3]
    y[:, 0] = 5.3578547 * x[:, 2]**2 + 0.8356891 * \
        x[:, 0] * x[:, 4] + 37.293239 * x[:, 0] - 40792.141
    y[:, 1] = -u
    y[:, 2] = u - 92
    y[:, 3] = -v + 90
    y[:, 4] = v - 110
    y[:, 5] = -w + 20
    y[:, 6] = w - 25

    return y


def G5MOD(x):
    """
    G5MOD
    binf = [0]
    binf.append(0)
    binf.append(-0.55)
    binf.append(-0.55)

    bsup = [1200]
    bsup.append(1200)
    bsup.append(0.55)
    bsup.append(0.55)
    """

    n, dim = x.shape
    if dim != 4:
        print("dimension must be equal 4")
        raise

    y = np.zeros((n, 6))

    y[:, 0] = 3 * x[:, 0] + 1e-6 * x[:, 0]**3 + \
        2 * x[:, 1] + (2 * 1e-6 / 3.) * x[:, 1]**3
    y[:, 1] = x[:, 2] - x[:, 3] - 0.55
    y[:, 2] = x[:, 3] - x[:, 2] - 0.55
    y[:, 3] = 1000 * np.sin(-x[:, 2] - 0.25) + 1000 * \
        np.sin(-x[:, 3] - 0.25) + 894.8 - x[:, 0]
    y[:, 4] = 1000 * np.sin(x[:, 2] - 0.25) + 1000 * \
        np.sin(x[:, 2] - x[:, 3] - 0.25) + 894.8 - x[:, 1]
    y[:, 5] = 1000 * np.sin(x[:, 3] - 0.25) + 1000 * \
        np.sin(x[:, 3] - x[:, 2] - 0.25) + 1294.8

    return y, {}


def G9(x):
    """
    G9
    binf = [-10]
    bsup = [10]
    for i in range(dim):
        binf.append(-10)
        bsup.append(10)
    """

    n, dim = x.shape
    if dim != 7:
        print("dimension must be equal 7")
        raise

    y = np.zeros((n, 5))
    y[:, 0] = (x[:, 0] - 10)**2 + 5 * (x[:, 1] - 12)**2 + x[:, 2]**4 + 3 * (x[:, 3] - 11)**2 + 10 * \
        x[:, 4]**6 + 7 * x[:, 5]**2 + x[:, 6]**4 - 4 * \
        x[:, 5] * x[:, 6] - 10 * x[:, 5] - 8 * x[:, 6]
    y[:, 1] = (2 * x[:, 0]**2 + 3 * x[:, 1]**4 + x[:, 2] +
               4 * x[:, 3]**2 + 5 * x[:, 4] - 127) / 127.
    y[:, 2] = (7 * x[:, 0] + 3 * x[:, 1] + 10 * x[:, 2]
               ** 2 + x[:, 3] - x[:, 4] - 282) / 282.
    y[:, 3] = (23 * x[:, 0] + x[:, 1]**2 + 6 *
               x[:, 5]**2 - 8 * x[:, 6] - 196) / 196.
    y[:, 4] = 4 * x[:, 0]**2 + x[:, 1]**2 - 3 * x[:, 0] * \
        x[:, 1] + 2 * x[:, 2]**2 + 5 * x[:, 5] - 11 * x[:, 6]
    return y, {}


def G10(x):
    """
    G10
    binf = [100]
    binf.append(1000)
    binf.append(1000)
    binf.append(10)
    binf.append(10)
    binf.append(10)
    binf.append(10)
    binf.append(10)

    bsup = [1e4]
    bsup.append(1e4)
    bsup.append(1e4)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    bsup.append(1e3)
    """

    n, dim = x.shape
    if dim != 8:
        print("dimension must be equal 8")
        raise

    y = np.zeros((n, 7))
    y[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
    y[:, 1] = -1 + 0.0025 * (x[:, 3] + x[:, 5])
    y[:, 2] = -1 + 0.0025 * (-x[:, 3] + x[:, 4] + x[:, 6])
    y[:, 3] = -1 + 0.01 * (-x[:, 4] + x[:, 7])
    for i in range(n):
        z = 100 * x[i, 0] - x[i, 0] * x[i, 5] + 833.33252 * x[i, 3] - 83333.333
        if z >= 0:
            y[i, 4] = np.log(1 + z)
        else:
            y[i, 4] = - np.log(1 - z)

        z = x[i, 1] * x[i, 3] - x[i, 1] * \
            x[i, 6] - 1250 * x[i, 3] + 1250 * x[i, 4]
        if z >= 0:
            y[i, 5] = np.log(1 + z)
        else:
            y[i, 5] = - np.log(1 - z)

        z = x[i, 2] * x[i, 4] - x[i, 2] * x[i, 7] - 2500 * x[i, 4] + 1250000
        if z >= 0:
            y[i, 6] = np.log(1 + z)
        else:
            y[i, 6] = - np.log(1 - z)

    return y, {}


def Hesse(x):
    """
    Hesse
    binf = [0]
    binf.append(0)
    binf.append(1)
    binf.append(0)
    binf.append(1)
    binf.append(0)

    bsup = [5]
    bsup.append(4)
    bsup.append(5)
    bsup.append(6)
    bsup.append(5)
    bsup.append(10)
    """

    n, dim = x.shape
    if dim != 6:
        print("dimension must be equal 6")
        raise

    y = np.zeros((n, 7))
    y[:, 0] = -25 * (x[:, 0] - 2)**2 - (x[:, 1] - 2)**2 - (x[:, 2] -
                                                           1)**2 - (x[:, 3] - 4)**2 - (x[:, 4] - 1)**2 - (x[:, 5] - 4)**2
    y[:, 1] = (2 - x[:, 0] - x[:, 1]) / 2.
    y[:, 2] = (x[:, 0] + x[:, 1] - 6) / 6.
    y[:, 3] = (-x[:, 0] + x[:, 1] - 2) / 2.
    y[:, 4] = (x[:, 0] - 3 * x[:, 1] - 2) / 2.
    y[:, 5] = (4 - (x[:, 2] - 3)**2 - x[:, 3]) / 4.
    y[:, 6] = (4 - (x[:, 4] - 3)**2 - x[:, 5]) / 4.

    return y, {}


def G18(x):
    """
    binf =[-10]
    bsup = [10]
    for i in range(7):
        binf.append(-10)
        bsup.append(10)

    binf.append(0)
    bsup.append(20)
    """

    n, dim = x.shape
    if dim != 9:
        print("dimension must be equal 9")
        raise

    y = np.zeros((n, 14))

    y[:, 0] = -0.5 * (x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2] + x[:, 2] * x[:, 8] - x[:, 4] * x[:, 8]
                      + x[:, 4] * x[:, 7] - x[:, 5] * x[:, 6])
    y[:, 1] = x[:, 2]**2 + x[:, 3]**2 - 1
    y[:, 2] = x[:, 8]**2 - 1
    y[:, 3] = x[:, 4]**2 + x[:, 5]**2 - 1
    y[:, 4] = x[:, 0]**2 + (x[:, 1] - x[:, 8])**2 - 1
    y[:, 5] = (x[:, 0] - x[:, 4])**2 + (x[:, 1] - x[:, 5])**2 - 1
    y[:, 6] = (x[:, 0] - x[:, 6])**2 + (x[:, 1] - x[:, 7])**2 - 1
    y[:, 7] = (x[:, 2] - x[:, 4])**2 + (x[:, 3] - x[:, 5])**2 - 1
    y[:, 8] = (x[:, 2] - x[:, 6])**2 + (x[:, 3] - x[:, 7])**2 - 1
    y[:, 9] = x[:, 6]**2 + (x[:, 7] - x[:, 8])**2 - 1
    y[:, 10] = x[:, 1] * x[:, 2] - x[:, 0] * x[:, 3]
    y[:, 11] = -x[:, 2] * x[:, 8]
    y[:, 12] = x[:, 4] * x[:, 8]
    y[:, 13] = x[:, 5] * x[:, 6] - x[:, 4] * x[:, 7]
    return y, {}


def G19(x):
    """
    binf = [0]
    bsup = [10]
    for i in range(14):
        binf.append(0)
        bsup.append(10)
    """

    n, dim = x.shape
    if dim != 15:
        print("dimension must be equal 15")
        raise

    y = np.zeros((n, 6))
    a = np.array([[-16, 2, 0, 1, 0], [0, -2, 0, 0.4, 2], [-3.5, 0, 2, 0, 0], [0, -2, 0, -4, -1],
                  [0, -9, -2, 1, -2.8], [2, 0, -4, 0, 0], [-1, -
                                                           1, -1, -1, -1], [-1, -2, -3, -2, -1],
                  [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])
    b = np.array([[-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]]).T
    c = np.array([[30, -20, -10, 32, -10], [-20, 39, -6, -31, 32], [-10, -6, 10, -6, -10],
                  [32, -31, -6, 39, -20], [-10, 32, -10, -20, 30]])
    d = np.array([[4, 8, 10, 6, 2]]).T
    e = np.array([[-15, -27, -36, -18, -12]]).T

    for k in range(n):
        s1 = 0
        s2 = 0
        s3 = 0
        for i in range(5):
            for j in range(5):
                s1 += c[i, j] * x[k, 10 + i] * x[k, 10 + j]

        for j in range(5):
            s2 += d[j] * x[k, 10 + j]**3

        for i in range(10):
            s3 += -b[i] * x[k, i]

        y[k, 0] = s1 + s2 + s3

        for j in range(5):
            s4 = 0
            s5 = 0
            for i in range(5):
                s4 += -2 * c[i, j] * x[k, 10 + i]

            for i in range(10):
                s5 += a[i, j] * x[k, i]

            y[k, j + 1] = s4 - e[j] + s5

    return y, {}

testSet = [(Carre, 2), (Rastrigin, 2), (Rosenbrock, 2), (Styblinski, 2),
           (Carre, 3), (Rastrigin, 3), (Rosenbrock, 3), (Styblinski, 3),
           (Carre, 4), (Rastrigin, 4), (Rosenbrock, 4), (Styblinski, 4),
           (Carre, 5), (Rastrigin, 5), (Rosenbrock, 5), (Styblinski, 5),
           (Carre, 6), (Rastrigin, 6), (Rosenbrock, 6), (Styblinski, 6),
           (Carre, 7), (Rastrigin, 7), (Rosenbrock, 7), (Styblinski, 7),
           (Carre, 8), (Rastrigin, 8), (Rosenbrock, 8), (Styblinski, 8),
           (Carre, 9), (Rastrigin, 9), (Rosenbrock, 9), (Styblinski, 9),
           (Carre, 10), (Rastrigin, 10), (Rosenbrock, 10), (Styblinski, 10),
           (Carre, 11), (Rastrigin, 11), (Rosenbrock, 11), (Styblinski, 11),
           (Carre, 12), (Rastrigin, 12), (Rosenbrock, 12), (Styblinski, 12),
           (Carre, 13), (Rastrigin, 13), (Rosenbrock, 13), (Styblinski, 13),
           (Carre, 14), (Rastrigin, 14), (Rosenbrock, 14), (Styblinski, 14),
           (Carre, 15), (Rastrigin, 15), (Rosenbrock, 15), (Styblinski, 15),
           (Carre, 16), (Rastrigin, 16), (Rosenbrock, 16), (Styblinski, 16),
           (Carre, 17), (Rastrigin, 17), (Rosenbrock, 17), (Styblinski, 17),
           (Carre, 18), (Rastrigin, 18), (Rosenbrock, 18), (Styblinski, 18),
           (Carre, 19), (Rastrigin, 19), (Rosenbrock, 19), (Styblinski, 19),
           (Carre, 20), (Rastrigin, 20), (Rosenbrock, 20), (Styblinski, 20),
           (Carre, 21), (Rastrigin, 21), (Rosenbrock, 21), (Styblinski, 21),
           (Carre, 22), (Rastrigin, 22), (Rosenbrock, 22), (Styblinski, 22),
           (Carre, 23), (Rastrigin, 23), (Rosenbrock, 23), (Styblinski, 23),
           (Carre, 24), (Rastrigin, 24), (Rosenbrock, 24), (Styblinski, 24),
           (Carre, 25), (Rastrigin, 25), (Rosenbrock, 25), (Styblinski, 25),
           (Carre, 26), (Rastrigin, 26), (Rosenbrock, 26), (Styblinski, 26),
           (Carre, 27), (Rastrigin, 27), (Rosenbrock, 27), (Styblinski, 27),
           (Carre, 28), (Rastrigin, 28), (Rosenbrock, 28), (Styblinski, 28),
           (Carre, 29), (Rastrigin, 29), (Rosenbrock, 29), (Styblinski, 29),
           (Carre, 30), (Rastrigin, 30), (Rosenbrock, 30), (Styblinski, 30),
           (Carre, 31), (Rastrigin, 31), (Rosenbrock, 31), (Styblinski, 31),
           (Carre, 32), (Rastrigin, 32), (Rosenbrock, 32), (Styblinski, 32)]
