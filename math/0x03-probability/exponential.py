#!/usr/bin/env python3
""" Initialize Exponential """


class Exponential:
    """ exponential class """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Initialization of the exponential class"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """ Calculates the value of the PDF """
        if x < 0:
            return 0
        return self.lambtha * (Exponential.e ** (self.lambtha * (-1) * x))

    def cdf(self, x):
        """ Calculates the value of the CDF """
        if x < 0:
            return 0
        return 1 - (Exponential.e ** (self.lambtha * -1 * x))
