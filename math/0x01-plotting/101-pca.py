#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
#import pickle as pl


#with open("pca.npz", 'rb') as handle:
#     my_array = pl.load(handle)
#lib = np.array(my_array)

lib = np.loadtxt("bezdekIris.data", delimiter=",")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

print(data_means)
print(norm_data)
