#!/usr/bin/env python3
""" Initialize Normal """


class Normal:
    """ exponential class """

    e = 2.7182818285
    pi = 3.1415926536

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

    def z_score(self, x):
        """ calculates z score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ calculates x """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates the value of the PDF """
        tmp = 1 / (self.stddev * ((2 * Normal.pi)) ** 0.5)
        tmp2 = Normal.e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return tmp * tmp2

    def cdf(self, x):
        """ Calculates the value of the CDF """
        if x < 0:
            return 0
        return 1 - (Exponential.e ** (self.lambtha * -1 * x))
