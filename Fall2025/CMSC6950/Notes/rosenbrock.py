import numpy as np


def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)


def drosenbrock(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def Hrosenbrock(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1], 1) - np.diag(400*x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H
