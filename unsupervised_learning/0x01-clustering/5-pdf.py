#!/usr/bin/env python3
""" PDF """
import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function
        of a Gaussian distribution:

        - X is a numpy.ndarray of shape (n, d) containing
            the data points whose PDF should be evaluated
        - m is a numpy.ndarray of shape (d,) containing the
            mean of the distribution
        - S is a numpy.ndarray of shape (d, d) containing the
            covariance of the distribution

        Returns: P, or None on failure
            - P is a numpy.ndarray of shape (n,) containing the
            PDF values for each data point
            All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    part_1 = np.sqrt((2*np.pi)**d * det)
    fac = np.einsum('...k,kl,...l->...', X-m, inv, X-m)
    pdf = np.exp(-fac / 2) / part_1
    pdf = np.where(pdf < 1e-300, 1e-300, pdf)
    return pdf
