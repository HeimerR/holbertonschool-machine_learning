#!/usr/bin/env python3
""" Mean and Covariance """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set
        - X is a numpy.ndarray of shape (n, d) containing the data set:
            - n is the number of data points
            - d is the number of dimensions in each data point
        If X is not a 2D numpy.ndarray, raise a TypeError with the message:
        "X must be a 2D numpy.ndarray"
        If n is less than 2, raise a ValueError with the message:
        "X must contain multiple data points"

        Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d) containing the mean of the
            data set
        cov is a numpy.ndarray of shape (d, d) containing the covariance
        matrix of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    print(X.shape[0])
    deviation = np.add(X, -mean)
    factor = 1/(X.shape[0])
    cov = np.matmul(X.T, deviation) * factor
    return mean, cov
