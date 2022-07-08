# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:10:06 2022

@author: ramra
"""

# The below implements Iterated
# Local Search Algorithm

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

seed(1)

# Stochastic Hill Climbing

def objective(v):
    x,y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20

# Check if point is in bounds

def in_bounds(point, bounds):
    
    for d in range(len(bounds)):
        
        if point[d] < bounds[d,0] or point[d] > bounds[d,1]:
            return False
        
    return True

# function for stochastic hill climbing

def hillclimbing(objective, bounds, n_iterations, step_size):
    
    solution = None
    
    while solution is None or not in_bounds(solution, bounds): 
    
        solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])

        solution_eval = objective(solution)
    
        for i in range(n_iterations):
            candidate = solution + randn(len(bounds))*step_size
        
            candidate_eval = objective(candidate)
        
            if candidate_eval <= solution_eval:
                solution, solution_eval = candidate, candidate_eval
            
                #print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
            
    return [solution, solution_eval] 


bounds = np.asarray([[-5.0,5.0], [-5.0, 5.0]])

n_iterations = 1000

step_size = 0.05

best, score = hillclimbing(objective, bounds, n_iterations, step_size)

print("Done!")

print('f(%s) = %.5f' %(best, score))

# Stochastic Hill Climbing with Random Restarts

def random_restarts(objective, bounds, n_iterations, step_size, n_restarts):
    
    best, best_eval = None, 1E+10
    
    for n in range(n_restarts):
        
        start_pt = None
        
        while start_pt is None or not in_bounds(start_pt, bounds): 
            
            start_pt = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])

            solution, solution_eval = hillclimbing(objective, bounds, n_iterations, step_size)
            
            if solution_eval < best_eval:
                
                best, best_eval = solution, solution_eval
                
                print('>%d f(%s) = %.5f' % (n, best, best_eval))
                
    return [best, best_eval] 

bounds = np.asarray([[-5.0,5.0], [-5.0, 5.0]])

n_iterations = 1000

step_size = 0.05

n_restarts = 30

best, score = random_restarts(objective, bounds, n_iterations, step_size, n_restarts)

print("Done!")

print('f(%s) = %.5f' %(best, score))

# Iterated Local Search Algorithm

def iterated_local_search(objective, bounds, n_iterations, step_size, n_restarts, p_size):
    
    best, best_eval = None, 1E+10
    
    for n in range(n_restarts):
        
        start_pt = None
        
        while start_pt is None or not in_bounds(start_pt, bounds): 
            
            start_pt = bounds[:,0] + randn(len(bounds))*p_size

            solution, solution_eval = hillclimbing(objective, bounds, n_iterations, step_size)
            
            if solution_eval < best_eval:
                
                best, best_eval = solution, solution_eval
                
                print('>%d f(%s) = %.5f' % (n, best, best_eval))
                
    return [best, best_eval] 

bounds = np.asarray([[-5.0,5.0], [-5.0, 5.0]])

n_iterations = 1000

step_size = 0.05

n_restarts = 30

p_size = 1.0

best, score = iterated_local_search(objective, bounds, n_iterations, step_size, n_restarts, p_size)

print("Done!")

print('f(%s) = %.5f' %(best, score))

