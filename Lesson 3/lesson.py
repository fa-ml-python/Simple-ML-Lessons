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