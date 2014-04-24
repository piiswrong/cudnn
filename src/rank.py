import numpy as np
import random

dim = 16
V = 100 
V1 = 70
V2 = 30
R1 = 2
p = 0.1

mat1 = np.dot(np.random.normal(0.0, 1.0, (V1, R1)), np.random.normal(0.0, 1.0, (R1, dim)))
mat2 = np.random.normal(0.0, 1.0, (V2, dim))
mat = np.concatenate((mat1, mat2), axis = 0)

def rank(M):
    u,s,v = np.linalg.svd(M)
    return np.sum( s > 1e-10 )

N = 10000
X = np.zeros((N, V), dtype = np.float32)
Y = np.zeros((N, ), dtype = np.float32)
for i in xrange(N):
    ind = [ j for j in xrange(V) if random.random() < p ]
    r = rank(mat[ind, :])
    print len(ind), r
    X[i, ind] = 1.0
    Y[i] = r

X.tofile('RankData.bin')
Y.tofile('RankLabel.bin')



