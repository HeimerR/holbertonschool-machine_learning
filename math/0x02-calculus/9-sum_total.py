#!/usr/bin/env python3
""" sum with limits """


def summation_i_squared(n):
    """ calculates sum(i=0; n; i**2) """
    if type(n) != int:
        return None
    if n == 1:
        return 1
    if n < 0:
        return None
    return n**2 + summation_i_squared((n-1))
