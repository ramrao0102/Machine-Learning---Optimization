# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:48:14 2022

@author: ramra
"""

# Basic Gradient Descent Optimization

import numpy as np
from numpy import arange
from numpy.random import seed
from numpy.random import rand
from matplotlib import pyplot

seed(1)

def objective(x):
    
    return x **2.0

def derivative(x):

    return x * 2.0

# implement gradient descent algorithm

def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    
    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
    
    for i in range(n_iter):
        
        gradient = derivative(solution)
        
        solution = solution - step_size*gradient
        
        solution_eval = objective(solution)
        
        print('>%i f(%s) = %.5f' % (i, solution, solution_eval))
       
    return [solution, solution_eval]

n_iter = 100

step_size = 0.1

bounds = np.asarray([[-1.0, 1.0]])

best, score = gradient_descent(objective, derivative, bounds, n_iter, step_size)

print('f(%s) = %.5f' %(best, score))

