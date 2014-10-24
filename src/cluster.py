import numpy as np

n_centers = 10
n_dim = n_centers 
n_samples = 200

sigma = 0.1

centers = np.eye(n_centers)
data = []
for i in xrange(n_centers):
    data.append(np.random.normal(0,sigma,(n_samples, n_dim)) + centers[i])

data.append(np.random.normal(0,1,(n_samples*n_centers, n_dim)))

data = np.concatenate(data, axis = 0)
Y = np.zeros((n_samples*n_centers*2,1))
Y[:n_samples*n_centers] = 1
data = np.concatenate((data, Y), axis=1)
np.random.shuffle(data)

N = data.shape[0]
X_train = data[:N/2,:-1]
Y_train = data[:N/2,-1]
X_test = data[N/2:,:-1]
Y_test = data[N/2:,-1]
X_train.astype(np.float32).tofile("../data/clusterData.train")
Y_train.astype(np.int32).tofile("../data/clusterLabel.train")
X_test.astype(np.float32).tofile("../data/clusterData.test")
Y_test.astype(np.int32).tofile("../data/clusterLabel.test")

fspec = open("../data/cluster_data", "w")
fspec.write(
"""
input-dim=%d
output-dim=1
train-data=clusterData.train
train-label=clusterLabel.train
train-items=%d
test-data=clusterData.test
test-label=clusterLabel.test
test-items=%d
"""%(X_train.shape[1], X_train.shape[0], X_test.shape[0]))
fspec.close()


