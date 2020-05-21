#!/usr/bin/env python3
""" Maximization """
import numpy as np


def maximization(X, g):
    """ calculates the maximization step in the EM algorithm for a GMM:

        - X is a numpy.ndarray of shape (n, d) containing the data set
        - g is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster

        Returns: pi, m, S, or None, None, None on failure
            - pi is a numpy.ndarray of shape (k,) containing the updated
                priors for each cluster
            - m is a numpy.ndarray of shape (k, d) containing the updated
                centroid means for each cluster
            - S is a numpy.ndarray of shape (k, d, d) containing the updated
                covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if n != g.shape[1]:
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for ki in range(k):
        den = np.sum(g[ki])
        pi[ki] = den / n
        m[ki] = np.sum(np.matmul(g[ki].reshape(1, n), X), axis=0) / den
        dif = (X - m[ki])
        S[ki] = np.dot(g[ki].reshape(1, n) * dif.T, dif) / den
    return pi, m, S
