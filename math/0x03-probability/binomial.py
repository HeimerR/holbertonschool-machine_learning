#!/usr/bin/env python3
""" Initialize Binomial """


class Binomial:
    """  class binomial """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """Initialization of the binomial class"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            stddev = (sum([(data[i] - mean) ** 2 for i in range(len(data))]) /
                      len(data))
            self.p = 1 - (stddev / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """ Calculates the value of the PMF """
        if type(k) != int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i
        k_fact = 1
        for j in range(1, k + 1):
            k_fact *= j
        tmp_fact = 1
        for m in range(1, (self.n - k) + 1):
            tmp_fact *= m
        tmp2 = n_fact / (k_fact * tmp_fact)
        return tmp2 * ((self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """ Calculates the value of the CDF """
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(k + 1)])
