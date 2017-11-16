# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:13:31 2017

@author: mvkoroteev
"""

import scipy as sp
import pandas
import matplotlib.pyplot as plt

""" Импорт исходных данных из файлов """

dj = pandas.read_csv("data/D&J-IND_101001_171001.txt")
gasp = pandas.read_csv("data/GAZP_101001_171001.txt")
yndx = pandas.read_csv("data/YNDX_101001_171001.txt")

"""
Модели линейной регрессии:
    Парная - y_ = b0 + b1*x
    Множественная - y_ = b0 + b1*x1 + b2*x2

Функция ошибки:
    J = (1/2m) * sum((y_ - y)**2)
    J = (1/2m) * sum((b0 + b1*x1 + b2*x2 - y)**2)
    
Частные производные (градиент) функции ошибки:
    dJ/db0 = (1/m) * sum(y_ - y)
    dJ/dbi = (1/m) * sum((y_ - y) * xi)
"""

""" Предварительная обработка данных """
res = pandas.merge(dj, gasp, on='<DATE>', suffixes=['_DJ', '_GASP'])
res1 = pandas.merge(res, yndx, on='<DATE>', suffixes=['_1', '_YNDX'])
## TODO разобраться с суффиксами при множественном слиянии
y = res1['<CLOSE>_DJ']
x1 = res1['<CLOSE>_GASP']
x2 = res1['<CLOSE>']

print("Initial data")
plt.scatter(x1, y)
plt.show()
plt.scatter(x2, y)
plt.show()


"""Нормализация данных """
x1min, x1max, ymin, ymax = min(x1), max(x1), min(y), max(y)
x2min, x2max = min(x2), max(x2)
x1 = (x1 - min(x1)) / (max(x1) - min(x1))
x2 = (x2 - min(x2)) / (max(x2) - min(x2))
y = (y - min(y)) / (max(y) - min(y))

#print("Normalized data with regression")
#plt.scatter(x, y)


class hypothesis(object):
    """ Класс, отвечающий за обучение парной линейной регрессии """
    def __init__(self):
        """ Параметры регрессии """
        self.theta = sp.array([0, 0])
    def apply(self, X):
        """ Метод, возвращающий теоретический результат 
        по переданным значениям факторов """
        return self.theta[0] + self.theta[1] * X
    def error(self, X, Y):
        """ Функция ошибки """
        return sum((self.apply(X) - Y)**2) / (2 * len(Y))
    def gradient_descent(self, x, y, alpha=0.7):
        """ Метод, реализующий градиентный спуск """
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
        return (steps, errors)

# TODO реализовать универсальный класс для множественной регресии

class hypothesis2(object):
    def __init__(self):
        """ Параметры регрессии """
        self.theta = sp.array([0, 0, 0])
    def apply(self, X1, X2):
        return self.theta[0] + self.theta[1] * X1 + self.theta[2] * X2
    def error(self, X1, X2, Y):
        """ Метод, возвращающий теоретический результат 
        по переданным значениям факторов """
        return sum((self.apply(X1, X2) - Y)**2) / (2 * len(Y))
    def gradient_descent(self, X1, X2, Y):
        """ Метод, реализующий градиентный спуск """
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
    
            theta0 = self.theta[0] - alpha * dJ0
            theta1 = self.theta[1] - alpha * dJ1
            theta2 = self.theta[2] - alpha * dJ2
            self.theta = sp.array([theta0, theta1, theta2])
            
            J = hyp.error(X1, X2, Y)
            
            steps.append(i)
            errors.append(J)
            
            i += 1
        return (steps, errors)

""" Заводим и обучаем нашу регрессию """
hyp = hypothesis2()

J = hyp.error(x1, x2, y)
print("Initial error:", J)

(steps, errors) = hyp.gradient_descent(x1, x2, y)
#plt.plot(x, y_, color="green")
#plt.show()

print("Final error:  ", errors[-1])
print("Final error difference:", errors[-1] - errors[-2])

plt.plot(steps, errors)
plt.show()

""" Денормализуем данные и параметры регрессии """
#print("Denormalized data with regression")
x1 = x1 * (x1max - x1min) + x1min
x2 = x2 * (x2max - x2min) + x2min
y = y * (ymax - ymin) + ymin
#plt.scatter(x, y)

theta1 = hyp.theta[1] * (ymax-ymin) / (x1max - x1min)
theta2 = hyp.theta[2] * (ymax-ymin) / (x2max - x2min)
theta0 = hyp.theta[0] * (ymax - ymin) + ymin - theta1*x1min - theta2*x1min
hyp.theta = sp.array([theta0, theta1, theta2])
J = hyp.error(x1, x2, y)
print("Denormalized data error", J)
#y_ = hyp.apply(x)
#plt.plot(x, y_, color="red")
#plt.show()
