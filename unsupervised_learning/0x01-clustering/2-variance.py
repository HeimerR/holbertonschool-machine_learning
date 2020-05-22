#!/usr/bin/env python3
""" Variance """
import numpy as np


def variance(X, C):
    """
        calculates the total intra-cluster variance for a data set:

        - X is a numpy.ndarray of shape (n, d) containing the data set
        - C is a numpy.ndarray of shape (k, d) containing the centroid
            means for each cluster

        Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(X.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if C.shape[0] >= X.shape[0]:
        return None

    try:
        n, d = X.shape
        k = C.shape[0]
        xi = np.tile(X, k).reshape(n, k, d)
        temp = C.reshape(-1)
        ci = np.tile(temp, (n, 1)).reshape(n, k, d)
        xc = xi-ci
        dist = np.linalg.norm(xc, axis=2)
        clss = np.min(dist, axis=1)
        variance = np.sum(clss**2)

        return np.sum(variance)
    except Exception:
        return None
