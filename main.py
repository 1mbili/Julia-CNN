import pylab as plt


x1= [0.0823,	    0.0649,	    0.0672,	    0.0431]
x2= [0.0296,	    0.0164,	    0.0177,	    0.0072]
x3= [0.0070,	    0.0053,	    0.0039,	    0.0032]
x4= [0.0029,	    0.0023,	    0.0015,	   8.9247e-4]
x5= [1.1085,	    0.9919,	    0.3763,	    0.2728]


y = ['gaz1', 'gaz2', 'gaz3', 'gaz4']
labels = [24
,54
,88
,157
,295]
plt.title("Wykres błędów względnych gazów w algorytmie SVD")
plt.plot(y, x1, label=labels[0])
plt.plot(y, x2, label=labels[1])
plt.plot(y, x3, label=labels[2])
plt.plot(y, x4, label=labels[3])
plt.plot(y, x5, label=labels[4])
plt.legend(title="liczba neuronów")
plt.grid()
plt.show()