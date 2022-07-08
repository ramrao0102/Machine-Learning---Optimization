
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:25:23 2022

@author: ramra
"""

# we implement stochastic hill climbing here

import numpy as np
from numpy import arange
from numpy.random import seed
from numpy.random import rand
from numpy.random import randn
from matplotlib import pyplot
from numpy import meshgrid

r_min, r_max = -5.0, 5.0

seed(1)

def objective(x):
    
    return x **2.0

sample = r_min + rand(10)*(r_max - r_min) 

sample_eval = objective(sample)

inputs = arange(r_min, r_max, 0.1)

def objective(x):
    
    return x **2.0

results = objective(inputs)

pyplot.plot(inputs, results)

pyplot.show()

seed(5)

n_iterations = 1000

step_size = 0.1

def hillclimbing(objective, bounds, n_iterations, step_size):
    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])

    solution_eval = objective(solution)
    
    scores = []
    
    solutions = []
    
    solutions.append(solution)
    
    scores.append(solution_eval)
    
    for i in range(n_iterations):
        candidate = solution + randn(len(bounds))*step_size
        
        candidate_eval = objective(candidate)
        
        if candidate_eval <= solution_eval:
            solution, solution_eval = candidate, candidate_eval
            
            scores.append(solution_eval)
            
            solutions.append(solution)
            
            print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
            
    return[solution, solution_eval, scores, solutions]

bounds = np.asarray([[-5.0, 5.0]])

best, score, scores, solutions = hillclimbing(objective, bounds, n_iterations, step_size)

print('f(%s) = %.5f' %(best, score))

pyplot.plot(scores, '.-')

pyplot.xlabel('Improvement No')

pyplot.ylabel('Evaluation f(x)')

pyplot.show()

evaluation = []

inputs = arange(bounds[0,0], bounds[0,1], 0.1)

for x in inputs:
    evaluation.append(objective(x))
    
inputs = arange(bounds[0,0], bounds[0,1], 0.1)

pyplot.plot(inputs, evaluation, '--')

pyplot.axvline(x = [0.0], ls = '--', color = 'red')

evaluation1 = []

for x in solutions:
    evaluation1.append(objective(x))
    
pyplot.plot(solutions, evaluation1, 'o', color = 'black')

pyplot.xlabel('Bounds Value')

pyplot.ylabel('Evaluation f(x)')

pyplot.show()
