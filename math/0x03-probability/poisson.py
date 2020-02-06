#!/usr/bin/env python3
""" Initialize Poisson """


class Poisson:
    """ prime class """
    def __init__(self, data=None, lambtha=1.):
        """Initialization of the poisson class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
