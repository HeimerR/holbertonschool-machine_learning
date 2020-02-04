#!/usr/bin/env python3
""" sum with limits """


def summation_i_squared(n):
    """ calculates sum(i=0; n; i**2) """
    if type(n) != int:
        return None
    if n <= 0:
        return 0
    return int((n * (n + 1) * (2 * n + 1)) / 6)
