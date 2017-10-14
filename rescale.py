import numpy as np

X = np.load('/home/danyang/mfs/data/hccr/image_1000x300x64x64_casia-offline.npy')
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		X[i,j] = X[i,j] / np.max(X[i,j])

np.save('/home/tongzheng/mfs/data/hccr/image_1000x300x64x64_casia-offline.npy', X)
