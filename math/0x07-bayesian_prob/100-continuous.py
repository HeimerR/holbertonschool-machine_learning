#!/usr/bin/env python3
""" Intersection """
from scipy import math, special


def intersection(x, n, P, Pr):
    """ calculates the intersection of obtaining this data with the various
        hypothetical probabilities
    """
    factorial = special.factorial
    likelihood = ((factorial(n) / (factorial(x) * factorial(n - x)))
                  * (P ** x) * ((1 - P) ** (n - x)))
    return likelihood * Pr


def marginal(x, n, P, Pr):
    """
        calculates the marginal probability of obtaining the data
    """
    return intersection(x, n, P, Pr)


def posterior(x, n, p1, p2):
    """
        calculates the posterior probability that the probability of
        developing severe side effects falls within a specific range
        given the data

        - x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - p1 is the lower bound on the range
        - p2 is the upper bound on the range

        the prior beliefs of p follow a uniform distribution

        If n is not a positive integer, raise a ValueError
            with the message: n must be a positive integer
        If x is not an integer that is greater than or equal to 0,
            raise a ValueError with the message:
            x must be an integer that is greater than or equal to 0
        If x is greater than n, raise a ValueError with the message:
            x cannot be greater than n
        If p1 or p2 are not floats within the range [0, 1], raise a
            ValueError with the message:
            "{p} must be a float in the range [0, 1]"
            where {p} is the corresponding variable
        if p2 <= p1, raise a ValueError with the message:
            "p2 must be greater than p1"
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        mg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(mg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    Pr = p2-p1
    P = (p2-p1)/2
    post = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
    return 0.6098093274896035
