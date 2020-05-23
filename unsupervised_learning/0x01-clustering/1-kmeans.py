#!/usr/bin/env python3
""" K-means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
        performs K-means on a dataset:

        - X is a numpy.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k is a positive integer containing the number of clusters

        iterations is a positive integer containing the maximum number
            of iterations that should be performed
        If no change occurs between iterations, your function should return
        Initialize the cluster centroids using a multivariate uniform
            distribution
        If a cluster contains no data points during the update step,
            reinitialize its centroid

        Returns: C, clss, or None, None on failure
        - C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        - clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
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
    C = np.random.uniform(low, high, (k, d))
    C_prev = np.copy(C)
    for i in range(iterations):
        xi = np.tile(X, k).reshape(n, k, d)
        temp = C.reshape(-1)
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
