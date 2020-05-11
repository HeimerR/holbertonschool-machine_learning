#!/usr/bin/env python3
""" Initialize """
import numpy as np


class MultiNormal:
    """ represents a Multivariate Normal distribution """

    def __init__(self, data):
        """
            - data is a numpy.ndarray of shape (d, n) containing the data set:
                - n is the number of data points
                - d is the number of dimensions in each data point

            If data is not a 2D numpy.ndarray, raise a TypeError
            with the message: "data must be a 2D numpy.ndarray"
            If n is less than 2, raise a ValueError
            with the message: "data must contain multiple data points"

            instance variables:
            mean - a numpy.ndarray of shape (d, 1)
                containing the mean of data
            cov - a numpy.ndarray of shape (d, d)
                containing the covariance matrix data
            You are not allowed to use the function numpy.cov
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        n = data.shape[1]
        d = data.shape[0]
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        deviation = np.tile(self.mean.reshape(-1), n).reshape(n, d)
        cov = data.T - deviation
        self.cov = np.matmul(cov.T, cov)/(n - 1)
