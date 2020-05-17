#!/usr/bin/env python3
""" Intersection """
import numpy as np


def intersection(x, n, P, Pr):
    """ calculates the intersection of obtaining this data with the various
        hypothetical probabilities
    """

    fact = np.math.factorial
    tmp = fact(n) / (fact(x) * fact(n - x))
    likelihood = tmp * (P**x) * ((1 - P)**(n - x))
    return likelihood * Pr


def marginal(x, n, P, Pr):
    """
        calculates the marginal probability of obtaining the data

        - x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects
        - Pr is a 1D numpy.ndarray containing the prior beliefs about P

        If n is not a positive integer, raise a ValueError
            with the message: n must be a positive integer
        If x is not an integer that is greater than or equal to 0,
            raise a ValueError with the message:
            x must be an integer that is greater than or equal to 0
        If x is greater than n, raise a ValueError with the message:
            x cannot be greater than n
        If P is not a 1D numpy.ndarray, raise a TypeError with the message:
            P must be a 1D numpy.ndarray
        If any value in P o Pr is not in the range [0, 1], raise a ValueError
            with the message: All values in {P/Pr} must be in the range [0, 1]
        If Pr does not sum to 1, raise a ValueError with the message:
            "Pr must sum to 1"
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        mg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(mg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.amin(Pr) < 0 or np.amax(Pr) > 1:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1])[0]:
        raise ValueError("Pr must sum to 1")
    return np.sum(intersection(x, n, P, Pr))
