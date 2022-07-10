# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:14:02 2022

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

seed(5)

# function to find the gradient descent for:

def objective(x,y):
    return x**2.0 + y**2.0

def derivative(x,y):
    return np.asarray ([x*2.0 , y*2.0])

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

yaxis = arange(bounds[1,0], bounds[1,1], 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

pyplot.contour(x ,y, results, 50, alpha = 1.0, cmap = 'jet')

pyplot.show() 

# Adam Algorithm

n_iterations = 100

alpha = 0.02

beta1 = 0.8

beta2 = 0.99

eps = 1.0E-08


def adam(bounds, objective, n_iterations, alpha, beta1, beta2, eps):

    solution = bounds[:,0] + rand(len(bounds))*(bounds[:,1] - bounds[:,0])
                       
    solution_eval = objective(solution[0] , solution[1])

    m = []
    
    solutions = []

    for _ in range(bounds.shape[0]):
        m.append(0.0)

    v = []

    for _ in range(bounds.shape[0]):
        v.append(0.0) 

    for it in range(n_iterations):

        g   = derivative(solution[0], solution[1])
        
        new_solution = []
                       
        for i in range(solution.shape[0]):
            
            m[i] = beta1 * m[i] + (1.0 - beta1)*g[i]

            v[i] = beta2 * v[i] + (1.0 - beta2)*g[i]**2.0

            mhat = m[i]/(1.0 - beta1**(it+1))

            vhat = v[i]/(1.0 - beta2**(it+1))
            
            value =  solution[i] - alpha *mhat/(sqrt(vhat) + eps)
            
            new_solution.append(value)
                      
        solution = np.asarray(new_solution)   
     
        solution_eval = objective(solution[0], solution[1])   

        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
        
        solutions.append(solution.copy())
             
    return [solution, solution_eval , solutions]

best, score , solutions = adam(bounds, objective, n_iterations, alpha, beta1, beta2, eps)
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

pyplot.plot(solutions[:,0], label='x-value')

pyplot.plot(solutions[:,1],  label='y-value')

pyplot.title('Convergence with Adam Optimization')
pyplot.xlabel('No of Iterations')
pyplot.ylabel('x- and y- values')

pyplot.legend()

pyplot.show()
                   