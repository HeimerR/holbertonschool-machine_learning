#!/usr/bin/env python3
""" K-means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
        performs K-means on a dataset:

    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) != int or k <= 0 or X.shape[0] <= k:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    n, d = X.shape
    C_prev = np.random.uniform(low, high, (k, d))
    C = np.zeros_like(C_prev)
    for i in range(iterations):
        xi = np.tile(X, k).reshape(n, k, d)
        temp = C_prev.reshape(-1)
        ci = np.tile(temp, (n, 1)).reshape(n, k, d)
        xc = xi-ci
        dist = np.linalg.norm(xc, axis=2)
        clss = np.argmin(dist, axis=1)
        for j in range(k):
            data_indx = np.where(clss == j)
            if len(data_indx[0]) == 0:
                C[j] = np.random.uniform(low, high, (1, d))
            else:
                C[j] = np.mean(X[data_indx], axis=0)
        if (C == C_prev).all():
            return C, clss
        C_prev = np.copy(C)
    return C, clss
