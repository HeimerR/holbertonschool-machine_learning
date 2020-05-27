#!/usr/bin/env python3
""" Absorbing Chains """
import numpy as np


def check(P, idx, n):
    """ check absorbing """
    abso = [False for i in range(n)]
    tmp = P[idx[0]]

    tmp = np.sum(tmp, axis=0)
    abso[idx[0][0]] = True

    col = P[:, idx[0][0]]
    for i in range(n):
        tmp2 = P[i] > 0
        tmp2 = tmp2 * tmp
        if (tmp2 == 1).any():
            tmp[i] = 1
    return tmp.all()


def absorbing(P):
    """  determines if a markov chain is absorbing:

        - P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
            - P[i, j] is the probability of transitioning from
                state i to state j
            - n is the number of states in the markov chain

        Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n = P.shape[0]
    if n != P.shape[1]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False

    d = np.diagonal(P)
    if (d == 1).all():
        return True
    if not (d == 1).any():
        return False
    idx = np.where(d == 1)
    return check(P, idx, n)
