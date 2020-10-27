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

    def pdf(self, x):
        """ x is a numpy.ndarray of shape (d, 1) containing the data point
            whose PDF should be calculated

                - d is the number of dimensions of the Multinomial instance

            If x is not a numpy.ndarray, raise a TypeError with the message:
            "x must by a numpy.ndarray"
            If x is not of shape (d, 1), raise a ValueError with the message:
            "x mush have the shape ({d}, 1)"

        Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # pdf formula -- multivar

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        f1 = 1 / np.sqrt(((2*np.pi)**d)*det)
        f21 = -(x-self.mean).T
        f22 = np.matmul(f21, inv)
        f23 = (x - self.mean) / 2
        f24 = np.matmul(f22, f23)
        f2 = np.exp(f24)
        pdf = f1 * f2

        return pdf.reshape(-1)[0]
