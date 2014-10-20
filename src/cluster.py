import numpy as np

n_centers = 10
n_dim = n_centers 
n_samples = 100

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

X = data[:,:-1]
Y = data[:,-1]
X.astype(np.float32).tofile("../data/clusterData.bin")
Y.astype(np.int32).tofile("../data/clusterLabel.bin")

fspec = open("../data/cluster_data", "w")
fspec.write(
"""
input-dim=%d
output-dim=1
train-data=clusterData.bin
train-label=clusterLabel.bin
train-items=%d
test-data=clusterData.bin
test-label=clusterLabel.bin
test-items=%d
"""%(X.shape[1], X.shape[0], X.shape[0]))
fspec.close()


