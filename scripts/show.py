import matplotlib.pyplot as plt
import numpy as np
import sys

x = np.loadtxt(sys.argv[1])
y = np.fromfile("../data/clusterLabel.test", dtype = np.int32)
plt.hold(True)
plt.scatter([ x[i,0] for i in xrange(x.shape[0]) if y[i] == 1 ], [ x[i,1] for i in xrange(x.shape[0]) if y[i] == 1 ], color = 'r') 
plt.scatter([ x[i,0] for i in xrange(x.shape[0]) if y[i] == 0 ], [ x[i,1] for i in xrange(x.shape[0]) if y[i] == 0 ], color = 'b') 
plt.savefig("a.png")
