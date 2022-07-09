# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:48:14 2022

@author: ramra
"""

# Basic Gradient Descent Optimization with Momemntum

import numpy as np
from numpy import arange
from numpy.random import seed
from numpy.random import rand
from matplotlib import pyplot

seed(15)

def objective(x):
    
    return x **2.0

def derivative(x):

    return x * 2.0

# implement gradient descent algorithm

def gradient_descent(objective, derivative, bounds, n_iter, step_size, momentun):
    
    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
    
    solutions = []
    
    scores = []
    
    change = 0.0
    
    for i in range(n_iter):
        
        gradient = derivative(solution)
        
        solution = solution - (step_size*gradient +momentum *change)
        
        solutions.append(solution)
        
        solution_eval = objective(solution)
        
        scores.append(solution_eval)
        
        change = step_size*gradient +momentum *change
        
        print('>%i f(%s) = %.5f' % (i, solution, solution_eval))
       
    return [solution, solution_eval, solutions, scores]

n_iter = 100

step_size = 0.1

momentum = 0.3

bounds = np.asarray([[-1.0, 1.0]])

best, score, solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)

print('f(%s) = %.5f' %(best, score))

inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)

results = objective(inputs)

pyplot.plot(inputs, results)

pyplot.plot(solutions, scores, '.-', color = 'red')

pyplot.show()