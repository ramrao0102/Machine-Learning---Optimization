# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:40:59 2022

@author: ramra
"""

from scipy.optimize import minimize
from numpy.random import rand

# the below implements BFGS Algorithm for Local Minima

def objective(x):
    return x[0]**2.0 + x[1]**2.0

def derivative(x):
    return [x[0]*2 , x[1]*2]

r_min, r_max = -5.0, 5.0

pt = r_min + rand(2) *(r_max-r_min)

result = minimize(objective, pt, method = 'L-BFGS-B', jac = derivative)

print('Status: %s' % result['message'])

print('Total Evaluations: %d' % result['nfev'])

solution = result['x']

evaluation = objective(solution)

# gets stuck on local optima

print('Solution: f(%s) = %.5f' % (solution, evaluation))

def objective(x):
    return x[0]**2.0 + x[1]**2.0 + x[2]**2.0 + x[3]**2.0

def derivative(x):
    return [x[0]*2 , x[1]*2, x[2]*2, x[3]*2]

r_min, r_max = -5.0, 5.0

pt = r_min + rand(4) *(r_max-r_min)

result = minimize(objective, pt, method = 'L-BFGS-B', jac = derivative)

print('Status: %s' % result['message'])

print('Total Evaluations: %d' % result['nfev'])

solution = result['x']

evaluation = objective(solution)

# gets stuck on local optima

print('Solution: f(%s) = %.5f' % (solution, evaluation))