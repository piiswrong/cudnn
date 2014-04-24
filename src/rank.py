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


for i in xrange(100):
    ind = [ i for i in xrange(V) if random.random() < p ]
    print len(ind), np.linalg.matrix_rank(mat[ind, :])


