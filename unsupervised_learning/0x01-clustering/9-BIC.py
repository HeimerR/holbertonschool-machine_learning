#!/usr/bin/env python3
""" BIC """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds the best number of clusters for a GMM using the Bayesian
            Information Criterion:

        - X is a numpy.ndarray of shape (n, d) containing the data set
        - kmin is a positive integer containing the minimum number of
            clusters to check for (inclusive)
        - kmax is a positive integer containing the maximum number of
            clusters to check for (inclusive)
        - iterations is a positive integer containing the maximum number
            of iterations for the EM algorithm
        - tol is a non-negative float containing the tolerance for the
            EM algorithm
        - verbose is a boolean that determines if the EM algorithm should
            print information to the standard output

        Returns: best_k, best_result, l, b, or None, None, None, None
            on failure
            - best_k is the best value for k based on its BIC
            - best_result is tuple containing pi, m, S
            - pi is a numpy.ndarray of shape (k,) containing the cluster
                priors for the best number of clusters
            - m is a numpy.ndarray of shape (k, d) containing the centroid
                means for the best number of clusters
            - S is a numpy.ndarray of shape (k, d, d) containing the
                covariance matrices for the best number of clusters
            - l_t is a numpy.ndarray of shape (kmax - kmin + 1) containing the
                log likelihood for each cluster size tested
            - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
                BIC value for each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
                - p is the number of parameters required for the model
                - n is the number of data points used to create the model
                - l is the log likelihood of the model
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None, None, None
        if type(kmin) != int or kmin <= 0 or X.shape[0] <= kmin:
            return None, None, None, None
        if type(kmax) != int or kmax <= 0 or X.shape[0] <= kmax:
            return None, None, None, None
        if type(iterations) != int or iterations <= 0:
            return None, None, None, None
        if type(tol) != float or tol < 0:
            return None, None, None, None
        if type(verbose) != bool:
            return None, None, None, None

        n, d = X.shape
        l_t = np.empty((kmax-kmin+1, ))
        b = np.empty((kmax-kmin+1, ))
        pi_t = []
        m_t = []
        S_t = []
        for k in range(kmin, kmax+1):
            pi, m, S, g, li = expectation_maximization(X,
                                                       k,
                                                       iterations,
                                                       tol,
                                                       verbose)
            if pi is None:
                return None, None, None, None
            pi_t.append(pi)
            m_t.append(m)
            S_t.append(S)
            l_t[k-1] = li
            p = (k * d * (d+1) / 2) + (d * k) + k - 1
            b[k-1] = p*np.log(n) - 2*li
        best_k = np.argmin(b)
        best_result = (pi_t[best_k], m_t[best_k], S_t[best_k])
        return best_k+1, best_result, l_t, b
    except Exception:
        return None, None, None, None
