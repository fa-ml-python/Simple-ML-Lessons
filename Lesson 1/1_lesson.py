import scipy as sp

""" Загружаем данные из файла """
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
#data[:10]
#data.shape

""" Разделяем данные на столбцы """
x = data[:,0]
y = data[:,1]

""" Обнаруживаем и удаляем отсутствующие данные """
#sp.sum(sp.isnan(y))
#Out[37]: 8
х = x[~sp.isnan(y)]
у = y[~sp.isnan(y)]


import matplotlib.pyplot as plt

plt.scatter(x, y, s=10)
plt.show()


import bokeh.plotting as bok

p = bok.figure(plot_width=800, plot_height=600)
p.circle(x, y)
bok.show(p)