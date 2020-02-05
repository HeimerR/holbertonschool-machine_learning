#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

print(pca_data)
print(len(pca_data.T[0]))
print(len(pca_data.T[1]))
print(len(pca_data.T[2]))
fig = plt.figure()
fig.suptitle("PCA of Iris Dataset")
ax = Axes3D(fig)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.plasma()
ax.scatter(pca_data.T[0], pca_data.T[1], pca_data.T[2], c=labels)
plt.show()
