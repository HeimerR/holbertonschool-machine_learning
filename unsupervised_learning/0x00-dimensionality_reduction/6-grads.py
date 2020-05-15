#!/usr/bin/env python3
""" Gradients """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """  calculates the gradients of Y
        - Y is a numpy.ndarray of shape (n, ndim) containing
        the low dimensional transformation of X

        - P is a numpy.ndarray of shape (n, n) containing
        the P affinities of X

        Returns: (dY, Q)
            - dY is a numpy.ndarray of shape (n, n) containing
            the gradients of Y
            - Q is a numpy.ndarray of shape (n, n) containing
            the Q affinities of Y
    """
    Q, num = Q_affinities(Y)
    n, m = Y.shape
    dY = np.zeros((n, m))
    for i in range(n):
        dY[i, :] = (np.sum(np.tile((P-Q)[:, i] * num[:, i],
                    (m, 1)).T * (Y[i, :] - Y), 0))
    return dY, Q
