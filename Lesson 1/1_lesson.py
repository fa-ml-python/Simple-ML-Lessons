import scipy as sp

""" Загружаем данные из файла """
data = sp.genfromtxt("data/web_traffic.tsv", delimiter="\t")
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
    from math import sqrt
    return sqrt(sp.sum((f(x) - y) ** 2) / len(x))

""" Строим модель парной линейной регрессии """

reg = sp.polyfit(x, y, deg=1)
f1 = sp.poly1d(reg)
print(reg, error(f1, x, y))
#fx = sp.linspace(0, x[-1], 1000)
#plt.scatter(x, y, s=10)
#plt.plot(fx, f1(fx), linewidth=4, color="red")
#plt.show()


""" Модель парной квадратичной регрессии """

reg = sp.polyfit(x, y, deg=2)
f2 = sp.poly1d(reg)
print(reg, error(f2, x, y))
#fx = sp.linspace(0, x[-1], 1000)
#plt.scatter(x, y, s=10)
#plt.plot(fx, f2(fx), linewidth=4, color="red")
#plt.show()


""" Регрессия 3 степени """

reg = sp.polyfit(x, y, deg=3)
f3 = sp.poly1d(reg)
print(reg, error(f3, x, y))
#fx = sp.linspace(0, x[-1], 1000)
#plt.scatter(x, y, s=10)
#plt.plot(fx, f3(fx), linewidth=4, color="red")
#plt.show()


""" Регрессия 10 степени """

reg = sp.polyfit(x, y, deg=10)
f10 = sp.poly1d(reg)
print(error(f10, x, y))
#fx = sp.linspace(0, x[-1], 1000)
#plt.scatter(x, y, s=10)
#plt.plot(fx, f10(fx), linewidth=4, color="red")
#plt.show()


""" Регрессия 100 степени """

reg = sp.polyfit(x, y, deg=100)
f100 = sp.poly1d(reg)
print(error(f100, x, y))
#fx = sp.linspace(0, x[-1], 1000)
#plt.scatter(x, y, s=10)
#plt.plot(fx, f100(fx), linewidth=4, color="red")
#plt.show()


""" Строим регрессию используя вспомогательную функцию """

from gists import plot_models

plot_models(x, y, [f1, f2, f3, f10], "train")



""" Разбиение выборки на тестовую и обучающую """

from gists import train_test_split

(train, test) = train_test_split(len(x), frac=0.8)
reg = sp.polyfit(x[train], y[train], deg=2)
f2_ = sp.poly1d(reg)
print(error(f2_, x[train], y[train]))
print(error(f2_, x[test], y[test]))

plot_models(x, y, [f2_], "test")