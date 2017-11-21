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

X = sp.array(data.data).transpose()
Y = sp.array(data.target)