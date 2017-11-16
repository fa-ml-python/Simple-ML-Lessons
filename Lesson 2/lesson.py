# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:13:31 2017

@author: mvkoroteev
"""

import scipy as sp
import pandas
import matplotlib.pyplot as plt



dj = pandas.read_csv("data/D&J-IND_101001_171001.txt")
gasp = pandas.read_csv("data/GAZP_101001_171001.txt")
yndx = pandas.read_csv("data/YNDX_101001_171001.txt")

"""
y = b0 + b1*x
y = b0 + b1*x1 + b2 * x2

"""

print(dj.shape, gasp.shape, yndx.shape)

#dj = dj[:gasp.shape[0]]

#x = gasp['<CLOSE>']
#y = dj['<CLOSE>']

res = pandas.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GASP'])
#print(res)
res1 = pandas.merge(res, yndx, on='<DATE>', suffixes=['_1', '_YNDX'])
#print(res1.shape)

y = res1['<CLOSE>_DJ']
x1 = res1['<CLOSE>_GASP']
x2 = res1['<CLOSE>']


x1min, x1max, ymin, ymax = min(x1), max(x1), min(y), max(y)
x2min, x2max = min(x2), max(x2)
x1 = (x1 - min(x1)) / (max(x1) - min(x1))
x2 = (x2 - min(x2)) / (max(x2) - min(x2))
y = (y - min(y)) / (max(y) - min(y))

#plt.scatter(x, y)


class hypothesis(object):
    def __init__(self):
        self.theta = sp.array([0, 0, 0])
    def apply(self, X1, X2):
        return self.theta[0] + self.theta[1] * X1 + self.theta[2] * X2
    def error(self, X1, X2, Y):
        return sum((self.apply(X1, X2) - Y)**2) / (2 * len(Y))
    def gradient_descent(self, X1, X2, Y):
        i = 0
        m = len(Y)
        alpha = 0.9
        steps = []
        errors = []
        while(i < 100):
            y_ = hyp.apply(X1, X2)
            dJ0 = sum(y_ - Y) / m
            dJ1 = sum((y_ - Y)*X1) / m
            dJ2 = sum((y_ - Y)*X2) / m
#            print(dJ0, dJ1)
    
            theta0 = self.theta[0] - alpha * dJ0
            theta1 = self.theta[1] - alpha * dJ1
            theta2 = self.theta[2] - alpha * dJ2
            self.theta = sp.array([theta0, theta1, theta2])
        #    print(hyp.theta)
            
            J = hyp.error(X1, X2, Y)
        #    print(J)
            
            steps.append(i)
            errors.append(J)
            
            i += 1
            
        return (steps, errors)


hyp = hypothesis()

#y_ = hyp.apply(x)
#plt.plot(x, y_, color="red")

J = hyp.error(x1, x2, y)
print(J)

(steps, errors) = hyp.gradient_descent(x1, x2, y)

#y_ = hyp.apply(x)
#plt.plot(x, y_, color="green")
#plt.show()

print(errors[-1])

plt.plot(steps, errors)
plt.show()



x1 = x1 * (x1max - x1min) + x1min
x2 = x2 * (x2max - x2min) + x2min
y = y * (ymax - ymin) + ymin
#plt.scatter(x, y)

theta1 = hyp.theta[1] * (ymax-ymin) / (x1max - x1min)
theta2 = hyp.theta[2] * (ymax-ymin) / (x2max - x2min)
theta0 = hyp.theta[0] * (ymax - ymin) + ymin - theta1*x1min - theta2*x1min
hyp.theta = sp.array([theta0, theta1, theta2])
J = hyp.error(x1, x2, y)
print(J)
#y_ = hyp.apply(x)
#plt.plot(x, y_, color="red")
#plt.show()