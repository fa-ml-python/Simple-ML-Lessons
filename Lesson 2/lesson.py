# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:13:31 2017

@author: mvkoroteev
"""

import scipy as sp
import pandas
import matplotlib.pyplot as plt



dj = pandas.read_csv("data/D&J-IND_101001_171001.txt")
yandex = pandas.read_csv("data/YNDX_101001_171001.txt")

dj = dj[:yandex.shape[0]]

x = yandex['<CLOSE>']
y = dj['<CLOSE>']

x = (x - min(x)) / (max(x) - min(x))
y = (y - min(y)) / (max(y) - min(y))

plt.scatter(x, y)



m = yandex.shape[0]
theta0 = 1
theta1 = 0

y_ = theta0 + theta1 * x
plt.plot(x, y_, color="red")

J = sum((y_ - y)**2) / (2*m)
print(J)

i = 0
steps = []
errors = []
while(i < 500):
    dJ0 = sum(y_ - y) / m
    dJ1 = sum((y_ - y)*x) / m
#    print(dJ0, dJ1)
    
    alpha = 0.2
    theta0 -= alpha * dJ0
    theta1 -= alpha * dJ1
    
    y_ = theta0 + theta1 * x
    
    J = sum((y_ - y)**2) / (2*m)
#    print(J)
    
    steps.append(i)
    errors.append(J)
    
    i += 1

plt.plot(x, y_, color="green")
plt.show()

plt.plot(steps, errors)
plt.show()