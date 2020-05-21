#!/usr/bin/env python3
""" EM """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM:

        - X is a numpy.ndarray of shape (n, d) containing the data set
        - k is a positive integer containing the number of clusters
        - iterations is a positive integer containing the maximum number of
            iterations for the algorithm
        - tol is a non-negative float containing tolerance of the log
            likelihood, used to determine early stopping i.e. if the
            difference is less than or equal to tol you should stop the
            algorithm
        - verbose is a boolean that determines if you should print information
            about the algorithm
            If True, print "Log Likelihood after {i} iterations: {l}" every 10
            iterations and after the last iteration
            {i} is the number of iterations of the EM algorithm
            {l} is the log likelihood

        Returns: pi, m, S, g, l, or None, None, None, None, None on failure
            - pi is a numpy.ndarray of shape (k,) containing the priors for
                each cluster
            - m is a numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster
            - S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for each cluster
            - g is a numpy.ndarray of shape (k, n) containing the probabilities
                for each data point in each cluster
            - l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] <= k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    n, d = X.shape
    l_prev = 0
    for i in range(iterations):
        g, li = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        if verbose is True and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, li))

        if abs(li - l_prev) <= tol:
            if verbose is True:
                print("Log Likelihood after {} iterations: {}".format(i, li))
            break
        l_prev = li

    return pi, m, S, g, li
