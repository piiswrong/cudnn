import matplotlib.pyplot as plt

fin = open('t.txt')
a = fin.readline().strip().split(' ')
y = a[0::3]
x1 = a[1::3]
x2 = a[2::3]
plt.hold(True)
plt.plot(y, x1)
plt.plot(y, x2)
plt.show()

