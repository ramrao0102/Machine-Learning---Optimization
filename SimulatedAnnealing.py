# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:52:55 2022

@author: ramra
"""

# This below code simulates
# simulated annealing algorithm

from numpy import arange
import numpy as np
from numpy import exp
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

temp = 10

def simulatedannealing(objective, bounds, n_iterations, step_size, temp):
    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])

    solution_eval = objective(solution)
    
    temperature = []
    
    for i in range(n_iterations):
        candidate = solution + randn(len(bounds))*step_size
        
        candidate_eval = objective(candidate)
        
        if candidate_eval <= solution_eval:
            solution, solution_eval = candidate, candidate_eval
            
            print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
          
        diff = candidate_eval - solution_eval
            
        t = temp/(float(i+1))
        
        temperature.append(t)
        
        metropolis = exp(-diff/t)
        
        if diff < 0 or rand() < metropolis:
            
            solution, solution_eval = candidate, candidate_eval
        
    return[solution, solution_eval, temperature]

bounds = np.asarray([[-5.0, 5.0]])

best, score, temperature = simulatedannealing(objective, bounds, n_iterations, step_size, temp)

print('f(%s) = %.5f' %(best, score))

print(temperature)

iterations = []

for i in range(n_iterations):
    iterations.append(i)
    
pyplot.plot(iterations, temperature) 

pyplot.xlabel('Iteration')
pyplot.ylabel('Temperature')

pyplot.show()   
