# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:35:59 2022

@author: ramra
"""

# Gradient Descent with RmsProp
# Algorithm Implementation

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

seed(5)

# function to find the gradient descent for:

def objective(x,y):
    return x**2.0 + y**2.0

# define input, create output, and plot ouput

r_min, r_max = -1.0, 1.0

xaxis = arange(r_min, r_max, 0.1)

yaxis = arange(r_min, r_max, 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

fig, ax = pyplot.subplots(subplot_kw = {"projection": "3d"})

ax.plot_surface(x,y, results, cmap = 'jet')

pyplot.show() 

# let us create a 2D plot

bounds = np.asarray([[-1.0,1.0], [-1.0, 1.0]])

xaxis = arange(bounds[0,0], bounds[0,1], 0.1)

xaxis = arange(bounds[1,0], bounds[1,1], 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

pyplot.contour(x ,y, results, 50, alpha = 1.0, cmap = 'jet')

pyplot.show() 

# Gradient Descent with RmsProp

step_size = 0.01

n_iterations =100

rho = 0.99

def objective(x,y):
    return x ** 2.0 + y ** 2.0

def derivative(x,y):
    return np.asarray([x*2.0, y*2.0])

bounds = np.asarray([[-1.0,1.0], [-1.0, 1.0]])


def rmsprop(objective, derivative, bounds, n_iterations, step_size, rho):

    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1]- bounds[:,0])

    sq_grad_avg = []
    
    solutions = []
    
    for it in range(n_iterations):

        for _ in range(bounds.shape[0]):
            sq_grad_avg.append(0.0)
    
        gradient = derivative(solution[0] , solution[1])    


        for i in range(gradient.shape[0]):
            sg = gradient[i]**2.0
    
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) +(sg *(1- rho))
    
        new_solution = []

        for i in range(solution.shape[0]):
            alpha = step_size/(1e-8 + sqrt(sq_grad_avg[i]))
            value = solution[i] - alpha*gradient[i]    
        
            new_solution.append(value)
    
        solution = np.asarray(new_solution)
        
        solutions.append(solution)
        
        solution_eval = objective(solution[0], solution[1])

        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
        
    return [solution, solution_eval, solutions]

best, score, solutions = rmsprop(objective, derivative, bounds, n_iterations, step_size, rho)

print('f(%s) = %.5f' %(best, score))

bounds = np.asarray([[-1.0,1.0], [-1.0, 1.0]])

xaxis = arange(bounds[0,0], bounds[0,1], 0.1)

yaxis = arange(bounds[1,0], bounds[1,1], 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

pyplot.contour(x ,y, results, 50, alpha = 1.0, cmap = 'jet')

solutions = np.asarray(solutions)

pyplot.plot(solutions[:,0], solutions[:,1], '.-', color = 'r')

pyplot.show() 
