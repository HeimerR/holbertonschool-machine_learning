#!/usr/bin/env python3
""" Regular Chains """
import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain:

        - P is a is a square 2D numpy.ndarray of shape (n, n) representing
            the transition matrix
            - P[i, j] is the probability of transitioning from state
                i to state j
            - n is the number of states in the markov chain

        Returns: a numpy.ndarray of shape (1, n) containing the steady state
            probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if n != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    P_init = np.copy(P)

    i = 2
    Ps = [P_init]
    while True:
        P = np.matmul(P_init, P)
        if any([(p == P).all() for p in Ps]) or i == 1000:
            return None
        if (P > 0).all():
            break
        Ps.append(P)
        i += 1
    P_init[np.diag_indices_from(P_init)] -= 1
    M = np.vstack((P_init.T[:-1], np.ones(n)))
    b = np.zeros(n)
    b[-1] = 1
    steady = np.linalg.solve(M, b).reshape(1, n)
    return steady
