import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.hold(True)
l = [2, 3, 5, 10]
for i in l:
    fin = open(str(i)+'.log')
    x = []
    y = []
    for line in fin:
        line = line.strip().split(' ')
        x.append(float(line[1]))
        y.append(float(line[0]))
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)

    for j in xrange(4):
        y[1:n-1] = (y[1:n-1] + y[2:])/2.0

    e = 0
    while e < n and x[e] < 1000:
        e += 1

    y = 100*(1-y)
    x = x
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')

    plt.plot(x[:e],y[:e], label='N=%d'%i)

plt.legend( loc = 'lower left' )
plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
matplotlib.rcParams.update({'font.size': 32})

plt.show()

