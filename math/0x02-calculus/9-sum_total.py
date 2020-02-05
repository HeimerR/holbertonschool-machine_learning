#!/usr/bin/env python3
""" sum with limits """


def summation_i_squared(n):
    """ calculates sum(i=0; n; i**2) """
    if type(n) != int or n <= 0:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
