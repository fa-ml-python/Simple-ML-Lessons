import scipy as sp

""" Загружаем данные из файла """
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
#data[:10]
#data.shape

""" Разделяем данные на столбцы """
x = data[:, 0]
y = data[:, 1]

""" Обнаруживаем и удаляем отсутствующие данные """
#sp.sum(sp.isnan(y))
#Out[37]: 8

x = x[~sp.isnan(y)].copy()
y = y[~sp.isnan(y)].copy()

""" Строим наши данные на графике """
import matplotlib.pyplot as plt

plt.scatter(x, y, s=10)
plt.show()


#import bokeh.plotting as bok
#
#p = bok.figure(plot_width=800, plot_height=600)
#p.circle(x, y)
#bok.show(p)


#import seaborn as sns
#
#sns.jointplot(x, y)

""" Функция ошибки """

def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

""" Строим модель парной линейной регрессии """

reg = sp.polyfit(x, y, deg=1)
f1 = sp.poly1d(reg)
print(reg, error(f1, x, y))
fx = sp.linspace(0, x[-1], 1000)
plt.scatter(x, y, s=10)
plt.plot(fx, f1(fx), linewidth=4, color="red")
plt.show()


""" Модель парной квадратичной регрессии """

reg = sp.polyfit(x, y, deg=2)
f1 = sp.poly1d(reg)
print(reg, error(f1, x, y))
fx = sp.linspace(0, x[-1], 1000)
plt.scatter(x, y, s=10)
plt.plot(fx, f1(fx), linewidth=4, color="red")
plt.show()