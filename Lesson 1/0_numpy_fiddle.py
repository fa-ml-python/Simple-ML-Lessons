import numpy as np


a = np.array([0, 1, 2, 3, 4, 5])
#a.ndim
#Out[6]: 1
#
#a.shape
#Out[7]: (6,)

b = a.reshape((3, 2))
#b
#Out[9]: 
#array([[0, 1],
#       [2, 3],
#       [4, 5]])
#
#b.ndim
#Out[10]: 2
#
#b.shape
#Out[11]: (3, 2)

b[1][0] = 77
#b
#Out[13]: 
#array([[ 0,  1],
#       [77,  3],
#       [ 4,  5]])
#
#a
#Out[14]: array([ 0,  1, 77,  3,  4,  5])

c = a.reshape((3,2)).copy()
c[0][0] = -99
#c
#Out[16]: 
#array([[-99,   1],
#       [ 77,   3],
#       [  4,   5]])
#
#a
#Out[17]: array([ 0,  1, 77,  3,  4,  5])

d = np.array([1, 2, 3, 4, 5])
#d*2
#Out[19]: array([ 2,  4,  6,  8, 10])
#d ** 2
#Out[20]: array([ 1,  4,  9, 16, 25], dtype=int32)


#a[[2, 3, 4]]
#Out[21]: array([77,  3,  4])
#a > 4
#Out[22]: array([False, False,  True, False, False,  True], dtype=bool)
#a[a>4]
#Out[23]: array([77,  5])
#a[a>4] = 4
#a
#Out[25]: array([0, 1, 4, 3, 4, 4])
#a.clip(0, 4)
#Out[26]: array([0, 1, 4, 3, 4, 4])

c = np.array([1, 2, np.NaN, 3, 4])
#c
#Out[28]: array([  1.,   2.,  nan,   3.,   4.])
#np.isnan(c)
#Out[29]: array([False, False,  True, False, False], dtype=bool)
#c[~np.isnan(c)]
#Out[31]: array([ 1.,  2.,  3.,  4.])