#!/usr/bin/env python3
""" Initialize Normal """


class Normal:
    """ exponential class """

    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialization of the normal class"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = ((sum([
                           (data[i] - self.mean) ** 2 for i in
                           range(len(data))])) / len(data)) ** 0.5

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
