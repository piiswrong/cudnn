import numpy as np
import os
import cv2

def makedata(name, path, list, patch_dims, dtype = np.float32):
    K = len(list)
    X = []
    Y = []
    for dirname,y in zip(list,xrange(K)):
        dirname = os.path.join(path,dirname)
        for filename in os.listdir(dirname):
            filename = os.path.join(dirname, filename)
            x = cv2.imread(filename, 1)
            x = cv2.resize(x, patch_dims)
            X.append(x)
            Y.append(y)
    X = np.asarray(X, dtype=dtype)
    Y = np.asarray(Y, dtype=np.int32)
    X = X.transpose((0,3,1,2)).reshape((X.shape[0], np.prod(X.shape[1:])))
    X -= np.mean(X, axis = 0)
    X /= np.std(X, axis = 0)
    p = np.random.permutation(X.shape[0])
    X = X[p]
    Y = Y[p]

    X.tofile('../data/%s_data.bin'%name)
    Y.tofile('../data/%s_label.bin'%name)
    with open('../data/%s'%name, 'w') as fspec:
        fspec.write(
"""input-dim=%d
input-dim=%d
input-dim=%d
output-dim=%d
yonehot=true
train-data=%s_data.bin
train-label=%s_label.bin
train-items=%d
test-data=%s_data.bin
test-label=%s_label.bin
test-items=%d
"""%(3,patch_dims[0],patch_dims[1], K, name, name, X.shape[0], name, name, X.shape[0]))


if __name__ == '__main__':
    path = '/s0/lavaniac/imageNet_train/'
    list = os.walk(path).next()[1][:10]
    print path, list
    makedata('imagenet', path, list, (227,227))

    







