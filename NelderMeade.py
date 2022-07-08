# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:02:23 2022

@author: ramra
"""

import numpy as np
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from matplotlib import pyplot
from numpy import meshgrid
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Now we will opimize non differentiable function
# with the Nelder Meade Algorithm

def objective(x):
    return x[0]**2.0 + x[1]**2.0

r_min, r_max = -5.0, 5.0

# the below is the starting point of the non differentiable function

pt = r_min + rand(2)*(r_max-r_min)

print(pt)

result = minimize(objective, pt, method='nelder-mead')

print('Status: %s' % result['message'])

print('Total Evaluations: %d' % result['nfev'])

solution = result['x']

evaluation = objective(solution)

print('Solution: f(%s) = %.5f' % (solution, evaluation))
      
# A noisy optimization problem

# The numpy.random.randn() function creates an array of specified shape and 
# fills it with random values as per standard normal distribution.

def objective(x):
    return (x + randn(len(x))*0.3)**2.0

r_min, r_max = -5.0, 5.0

# the below is the starting point of the non differentiable function
# this function does not converge to its optima

pt = r_min + rand(1)*(r_max-r_min)

print(pt)

result = minimize(objective, pt, method='nelder-mead')

print('Status: %s' % result['message'])

print('Total Evaluations: %d' % result['nfev'])

solution = result['x']

evaluation = objective(solution)

print('Solution: f(%s) = %.5f' % (solution, evaluation))

# Multimodal Optimization Problem

# Ackley Function

def objective(x,y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20

r_min, r_max = -5.0, 5.0

xaxis = arange(r_min, r_max, 0.1)

yaxis = arange(r_min, r_max, 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

fig, ax = pyplot.subplots(subplot_kw = {"projection": "3d"})

ax.plot_surface(x, y, results, cmap = 'jet')

pyplot.show()

def objective(v):
    x,y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20


pt = r_min + rand(2) *(r_max-r_min)

result = minimize(objective, pt, method='nelder-mead')

print('Status: %s' % result['message'])

print('Total Evaluations: %d' % result['nfev'])

solution = result['x']

evaluation = objective(solution)

# gets stuck on local optima

print('Solution: f(%s) = %.5f' % (solution, evaluation))

 


