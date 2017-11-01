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
х = x[-sp.isnan(y)]
у = y[-sp.isnan(y)]
