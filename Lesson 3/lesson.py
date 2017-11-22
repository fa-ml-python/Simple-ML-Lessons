# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:33:28 2017

@author: sejros
"""

import scipy as sp
import pandas
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

raw_features = sp.array(data.data).transpose()
raw_label = sp.array(data.target)


X = []
for feature in raw_features:
    feature = (feature - min(feature)) / (max(feature) - min(feature))
    X.append(feature)
X = sp.array(X)

Y = raw_label.copy()
Y[Y!=2] = 0
Y[Y==2] = 1

class hypothesis2(object):
    def __init__(self, n=2):
        """ Параметры регрессии """
        self.theta = [0] * (n+1)
    def apply(self, X):
        res = self.theta[0]
        for i in range(len(X)):
            res = self.theta[i+1] * X[i] + res
        return 1 / (1 + sp.exp(-res))
    def error(self, X, Y):
        """ Метод, возвращающий теоретический результат 
        по переданным значениям факторов """
        return sum((self.apply(X) - Y)**2) / (2 * len(Y))
    def gradient_descent(self, X, Y, n_steps=1000, test=False, 
                         x_test=None, y_test=None):
        """ Метод, реализующий градиентный спуск """
        i = 0
        m = len(Y)
        alpha = 0.3
        steps = []
        errors = []
        errors_test = []
        while(i < n_steps):
            y_ = self.apply(X)
    
            self.theta[0] = self.theta[0] - alpha * sum(y_ - Y) / m
            
            for j in range(len(X)):
                self.theta[j+1] = self.theta[j+1] - alpha * sum((y_ - Y)*X[j]) / m
            
            J = self.error(X, Y)
            
            steps.append(i)
            errors.append(J)
            
            if test:
                J_test = self.error(x_test, y_test)
                errors_test.append(J_test)
            
            i += 1
        if test:
            return (steps, errors, errors_test)
        else:
            return (steps, errors)
            

""" Заводим и обучаем нашу регрессию """
hyp = hypothesis2(n=4)

J = hyp.error(X, Y)
print("Initial error:", J)

(steps, errors) = hyp.gradient_descent(X, Y)
#plt.plot(x, y_, color="green")
#plt.show()

print("Final error:  ", errors[-1])

plt.plot(steps, errors)
plt.show()


""" Пробуем обучить регрессию на тренировочной выборке из данных """
  
def train_test_split(length, frac=0.9):
    split_idx = int(frac * length)
    shuffled = sp.random.permutation(list(range(length)))
    train = sorted(shuffled[:split_idx])
    test = sorted(shuffled[split_idx:])
    return (train, test)

hyp2 = hypothesis2(n=4)

(train, test) = train_test_split(len(Y), frac=0.8)

J = hyp2.error(X[:, train], Y[train])
print("Initial train error:", J)
J = hyp2.error(X[:, test], Y[test])
print("Initial test error:", J)

(steps, errors, errors_test) = hyp2.gradient_descent(X[:, train], Y[train], 
                                                    test=True, n_steps=2000,
                                                    x_test=X[:, test], y_test=Y[test])
print("Final train error:  ", errors[-1])
J = hyp2.error(X[:, test], Y[test])
print("Final test error:", J)
plt.plot(steps, errors)
plt.plot(steps, errors_test, color="purple")
plt.show()