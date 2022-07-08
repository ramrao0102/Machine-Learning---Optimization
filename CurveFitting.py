# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:53:35 2022

@author: ramra
"""

from pandas import read_csv
from numpy import arange
from numpy import sin
from numpy import sqrt
from scipy.optimize import curve_fit
from matplotlib import pyplot

# linear function

def objective(x, a, b):
    return a*x + b

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header = None)

data = dataframe.values

x,y = data[:,4], data[:,-1]

popt, _ = curve_fit(objective, x, y)

a, b = popt

print('y = %.5f * x + %.5f' %(a,b))

pyplot.scatter(x,y)

x_line = arange(min(x), max(x), 1)

y_line = objective(x_line, a,b)

pyplot.plot(x_line, y_line, ls = '--', color = 'red')

pyplot.show()

# non liner polynomial function

def objective(x, a, b, c):
    return a*x + b *x**2 + c

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header = None)

data = dataframe.values

x,y = data[:,4], data[:,-1]

popt, _ = curve_fit(objective, x, y)

a, b , c = popt

print('y = %.5f * x + %.5f * x^2 +  %.5f' %(a,b, c))

pyplot.scatter(x,y)

x_line = arange(min(x), max(x), 1)

y_line = objective(x_line, a,b, c)

pyplot.plot(x_line, y_line, ls = '--', color = 'red')

pyplot.show()

# 5th degree non linear polynomial function

def objective(x, a, b, c, d, e, f):
    return a*x + b *x**2 + c * x**3 + d * x**4 + e * x**5 + f

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header = None)

data = dataframe.values

x,y = data[:,4], data[:,-1]

popt, _ = curve_fit(objective, x, y)

a, b , c, d, e, f = popt

print('y = %.5f * x + %.5f * x^2 + + %.5f * x^3 + %.5f * x^4 +  %.5f * x^5 + %.5f' %(a,b, c, d, e, f))

pyplot.scatter(x,y)

x_line = arange(min(x), max(x), 1)

y_line = objective(x_line, a,b, c, d, e, f)

pyplot.plot(x_line, y_line, ls = '--', color = 'red')

pyplot.show()


# this one's interesting , a sine wave and a second degree ploynomial function

def objective(x, a, b, c, d):
    return a * sin(b-x) + c * x**2 + d

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'

dataframe = read_csv(url, header = None)

data = dataframe.values

x,y = data[:,4], data[:,-1]

popt, _ = curve_fit(objective, x, y)

a, b , c, d = popt

print('y = %.5f * sin(%.5f - x) + %.5f * x **2 + %.5f ' %(a,b, c, d))

pyplot.scatter(x,y)

x_line = arange(min(x), max(x), 1)

y_line = objective(x_line, a,b, c, d)

pyplot.plot(x_line, y_line, ls = '--', color = 'red')

pyplot.show()