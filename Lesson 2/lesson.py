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

<<<<<<< HEAD
print("Raw data shapes:", dj.shape, gasp.shape)
=======
"""
y = b0 + b1*x
y = b0 + b1*x1 + b2 * x2

"""

print(dj.shape, gasp.shape, yndx.shape)

#dj = dj[:gasp.shape[0]]

#x = gasp['<CLOSE>']
#y = dj['<CLOSE>']
>>>>>>> d768b5a40e45a802b51d34c7d07c072e9a24e789

res = pandas.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GASP'])
#print(res)
res1 = pandas.merge(res, yndx, on='<DATE>', suffixes=['_1', '_YNDX'])
#print(res1.shape)

y = res1['<CLOSE>_DJ']
x1 = res1['<CLOSE>_GASP']
x2 = res1['<CLOSE>']

<<<<<<< HEAD
print("Initial data")
plt.scatter(x, y)
plt.show()


minx, maxx, miny, maxy = min(x), max(x), min(y), max(y)
x = (x - min(x)) / (max(x) - min(x))
y = (y - min(y)) / (max(y) - min(y))

print("Normalized data with regression")
plt.scatter(x, y)
=======

x1min, x1max, ymin, ymax = min(x1), max(x1), min(y), max(y)
x2min, x2max = min(x2), max(x2)
x1 = (x1 - min(x1)) / (max(x1) - min(x1))
x2 = (x2 - min(x2)) / (max(x2) - min(x2))
y = (y - min(y)) / (max(y) - min(y))

#plt.scatter(x, y)
>>>>>>> d768b5a40e45a802b51d34c7d07c072e9a24e789


class hypothesis(object):
    def __init__(self):
<<<<<<< HEAD
        self.theta = sp.array([0, 0])
    def apply(self, X):
        return self.theta[0] + self.theta[1] * X
    def error(self, X, Y):
        return sum((self.apply(X) - Y)**2) / (2 * len(Y))
    def gradient_descent(self, x, y, alpha=0.7):
        i = 0
        steps = []
        errors = []
        m = len(y)
        while(i < 150):
            y_ = hyp.apply(x)
            dJ0 = sum(y_ - y) / m
            dJ1 = sum((y_ - y)*x) / m
            theta0 = self.theta[0] - alpha * dJ0
            theta1 = self.theta[1] - alpha * dJ1
            self.theta = sp.array([theta0, theta1])
            
            steps.append(i)
            errors.append(hyp.error(x, y))
            
            i += 1
=======
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
            
>>>>>>> d768b5a40e45a802b51d34c7d07c072e9a24e789
        return (steps, errors)


hyp = hypothesis()

#y_ = hyp.apply(x)
#plt.plot(x, y_, color="red")

<<<<<<< HEAD
J = hyp.error(x, y)
print("Initial error", J)

(steps, errors) = hyp.gradient_descent(x, y)
y_ = hyp.apply(x)
plt.plot(x, y_, color="green")
plt.show()
=======
J = hyp.error(x1, x2, y)
print(J)

(steps, errors) = hyp.gradient_descent(x1, x2, y)

#y_ = hyp.apply(x)
#plt.plot(x, y_, color="green")
#plt.show()
>>>>>>> d768b5a40e45a802b51d34c7d07c072e9a24e789

print("Final error difference:", errors[-1] - errors[-2])

plt.plot(steps, errors)
plt.show()

<<<<<<< HEAD
print("Denormalized data with regression")
x = x * (maxx - minx) + minx
y = y * (maxy - miny) + miny
theta1 = hyp.theta[1] * (maxy - miny) / (maxx - minx)
theta0 = hyp.theta[0] * (maxy - miny) + miny - theta1 * minx
print(theta0, theta1)
y__ = theta0 + theta1 * x
plt.scatter(x, y)
#plt.plot(x, y_, color="red")
plt.plot(x, y__, color="green")
plt.show()
=======


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
>>>>>>> d768b5a40e45a802b51d34c7d07c072e9a24e789
