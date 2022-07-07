
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:47:31 2022

@author: ramra
"""

import numpy as np
from numpy import arange
from numpy.random import seed
from numpy.random import rand
from matplotlib import pyplot
from numpy import meshgrid

from scipy.optimize import minimize_scalar

# convex univariate function optimization

def objective(x):
    return (5.0 + x) **2.0

r_min, r_max = -10.0, 10.0

inputs = arange(r_min, r_max, 0.1)

targets = [objective(x) for x in inputs]

pyplot.plot(inputs , targets, '--')  

pyplot.title('this is a convex function!')

result = minimize_scalar(objective, method = 'brent')

opt_x , opt_y = result['x'] , result['fun']

pyplot.plot([opt_x ], [opt_y], '*', color = 'red')

pyplot.show()

print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])

# non convex univariate function optimization

def objective(x):
    return (x - 2.0)*x*(x+2.0)**2.0

r_min, r_max = -3.0, 2.5

inputs = arange(r_min, r_max, 0.1)

targets = [objective(x) for x in inputs]

pyplot.plot(inputs , targets,  '--')  

pyplot.title('this is a non convex function!')

result = minimize_scalar(objective, method = 'brent')

opt_x , opt_y = result['x'] , result['fun']

pyplot.plot([opt_x ], [opt_y], '*', color = 'red')

pyplot.show()

print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
print('Total Evaluations n: %d' % result['nfev'])



