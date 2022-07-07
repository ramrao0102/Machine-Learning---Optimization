# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:35:29 2022

@author: ramra
"""

# let us write some math optimization problems

import numpy as np
from numpy import arange
from numpy.random import seed
from numpy.random import rand
from matplotlib import pyplot
from numpy import meshgrid
from mpl_toolkits.mplot3d import Axes3D

# Random Search

def objective(x):
    return x**2.0

r_min, r_max = -5.0, 5.0

sample = r_min + rand(100)*(r_max - r_min)

sample_eval = objective(sample)

best_ix = 0

for i in range(len(sample)):
    if (sample_eval[i]) < sample_eval[best_ix]:
        best_ix = i
        
print('Best:  f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))

inputs = arange(r_min, r_max, 0.1)

results = objective(inputs)

pyplot.plot(inputs, results)

pyplot.scatter(sample, sample_eval)


pyplot.axvline(x = sample_eval[best_ix], ls = '--', color = 'red')

pyplot.show()

# Grid Search

step = 0.1

sample = list()

for x in arange(r_min, r_max, step):
  
        sample.append(x)
        
# evaluate the sample

best_ix = 0

for i in range(len(sample)):
    sample_eval[i] = objective(sample[i])
    if sample_eval[i] <sample_eval[best_ix]:
        best_ix = i
        
print('Best:  f(%.5f) = %.5f' % (sample[best_ix], sample_eval[best_ix]))

# let us try a 2D Grid Search

def objective(x, y):
    return x**2.0 + y**2.0

r_min, r_max = -5.0, 5.0

step = 0.1

sample = list()

for x in arange(r_min, r_max, step):
    for y in arange(r_min, r_max, step):
        sample.append([x , y])
        
# evaluate the sample

best_ix = 0

sample_eval = []

for x,y in sample:
    sample_eval.append(objective(x,y))

      
for i in range(len(sample)):
    if sample_eval[i] < sample_eval[best_ix]:
        best_ix = i
        
print('Best: f(%.5f, %.5f) = %.5f' % (sample[best_ix][0], sample[best_ix][1], sample_eval[best_ix]))

# let us do a random search on 2D space


sample = list()

# let us put  list of x,y random_values

for x in arange(r_min, r_max, step):
    for y in arange(r_min, r_max, step):
        sample.append([r_min + rand()*(r_max-r_min), r_min + rand()*(r_max-r_min)])


# evaluate the sample

best_ix = 0

sample_eval = []

for x,y in sample:
    sample_eval.append(objective(x,y))

      
for i in range(len(sample)):
    if sample_eval[i] < sample_eval[best_ix]:
        best_ix = i
        
print('Best: f(%.5f, %.5f) = %.5f' % (sample[best_ix][0], sample[best_ix][1], sample_eval[best_ix]))

#printing results of objective function and optimal value on plot 

xaxis = arange(r_min, r_max, 0.1)

yaxis = arange(r_min, r_max, 0.1)

x,y = meshgrid(xaxis, yaxis)

results = objective(x,y)

pyplot.contour(x ,y, results, 50, alpha = 1.0, cmap = 'jet')

optima_x = [sample[best_ix][0], sample[best_ix][1]]

pyplot.plot([optima_x[0]], [optima_x[1]], '*', color = 'red')
        