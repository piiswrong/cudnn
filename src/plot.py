import matplotlib.pyplot as plt

plt.hold(True)
l = [2, 3]
for i in l:
    fin = open(str(i)+'.log')
    x = []
    y = []
    for line in fin:
        line = line.strip().split(' ')
        x.append(float(line[1]))
        y.append(float(line[0]))
    plt.plot(x,y)

plt.show()

