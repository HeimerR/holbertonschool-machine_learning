#!/usr/bin/env python3
""" EM """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM:

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
        if verbose:
            if i % 10 == 0 or i == iterations-1 or abs(li-l_prev) <= tol:
                print("Log Likelihood after {} iterations: {}".format(i, li))
        if abs(li-l_prev) <= tol:
            break
        l_prev = li

    return pi, m, S, g, li
