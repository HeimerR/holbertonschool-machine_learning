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

    def pmf(self, k):
        """ Calculates the value of the PMF """
        if type(k) != int:
            k = int(k)
        if k <= 0:
            return 0
        fac = 1
        for i in range(1, k + 1):
            fac *= i
        return ((self.lambtha ** k) * Poisson.e ** (self.lambtha * (-1))) / fac

    def cdf(self, k):
        """ Calculates the value of the CDF """
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(1, k + 1)])
