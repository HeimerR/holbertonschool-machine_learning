#!/usr/bin/env python3
""" sum with limits """


def summation_i_squared(n):
    """ calculates sum(i=0; n; i**2) """
    return sum([i**2 for i in range(1, n + 1)])
