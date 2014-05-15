import numpy as np
import random

def rankdata():
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
        if len(ind):
            r = rank(mat[ind, :])
            print len(ind), r
            X[i, ind] = 1.0
            Y[i] = 1.0*r/dim


    X.tofile('../data/RankData.bin')
    Y.tofile('../data/RankLabel.bin')


def graphdata():
    V = 4
    E = V*(V-1)/2
    edge = []
    for i in xrange(V):
        for j in xrange(i+1, V):
            edge.append((i,j))

    X = []
    Y = []
    for i in xrange(2**E):
        l = [ (i & (1 << j)) and 1 or 0 for j in xrange(E) ]
        if sum(l) < 3:
            y = sum(l)
        elif sum(l) == 3:
            mark = [ 0 for j in xrange(V) ]
            for j in xrange(E):
                if l[j]:
                    mark[edge[j][0]] = mark[edge[j][1]] = 1
            if sum(mark) == 3:
                y = 2
            else:
                y = 3
        else:
            y = 3
        X.append(l)
        Y.append(y/6.0)
    X = np.asarray(X, dtype = np.float32)
    Y = np.asarray(Y, dtype = np.float32)
    Y = np.resize(Y, (2**E,1))
    print X.shape, Y.shape
    print np.concatenate((X,Y), axis = 1)

    X.tofile('../data/RankData.bin')
    Y.tofile('../data/RankLabel.bin')
        
def dummydata():
    N = 64
    V = 6
    H = 16
    L = 2
    x = np.random.normal(size=(N,V))
    x = np.asarray(x, dtype=np.float32)
    x = x*(x>0)
    y = x
    fout = open("rank.param", "w")
    fout.write("%d\n"%L)
    for l in xrange(L):
        ni = no = H
        if l == L-1:
            no = 1
        if l == 0:
            ni = V
        w = np.random.normal(0, 1.0/np.sqrt(ni), size=(ni,no))
        w = np.asarray(w, dtype=np.float32)
        w = w*(w>0)
        fout.write("%d\ng%d %d %d\n"%(l,l,no,ni))
        for i in xrange(w.shape[0]):
            for j in xrange(w.shape[1]):
                fout.write(str(w[i,j])+" ")
            fout.write("\n")

        y = np.tanh(np.dot(y,w))
    x.tofile("../data/RankData.bin")
    y.tofile("../data/RankLabel.bin")
    print y


#rankdata()
#graphdata()
dummydata()
