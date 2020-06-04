#!/usr/bin/env python3
""" Initialize Bayesian Optimization """
import numpy as np
GP = __import__('2-gp').GaussianProcess
from scipy.stats import norm
from scipy.optimize import minimize


class BayesianOptimization:
    """ Bayesian optimization on a noiseless 1D Gaussian process:
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
            - f is the black-box function to be optimized
            - X_init is a numpy.ndarray of shape (t, 1) representing
                the inputs already sampled with the black-box function
            - Y_init is a numpy.ndarray of shape (t, 1) representing the
                outputs of the black-box function for each input in X_init
                - t is the number of initial samples
            - bounds is a tuple of (min, max) representing the bounds of the
                space in which to look for the optimal point
            - ac_samples is the number of samples that should be analyzed
                during acquisition
            - l is the length parameter for the kernel
            - sigma_f is the standard deviation given to the output of the
                black-box function
            - xsi is the exploration-exploitation factor for acquisition
            - minimize is a bool determining whether optimization should be
                performed for minimization (True) or maximization (False)

            instance attributes:
            - f: the black-box function
            - gp: an instance of the class GaussianProcess
            - X_s: a numpy.ndarray of shape (ac_samples, 1) containing all
                acquisition sample points, evenly spaced between min and max
            - xsi: the exploration-exploitation factor
            - minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        b_min, b_max = bounds
        self.X_s = np.linspace(b_min, b_max, num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize


    def acquisition(self):
        """ calculates the next best sample location:

            Uses the Expected Improvement acquisition function
            Returns: X_next, EI
                - X_next is a numpy.ndarray of shape (1,) representing the
                    next best sample point
                - EI is a numpy.ndarray of shape (ac_samples,) containing the
                    expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)


        if self.minimize is True:
            f_plus = np.amin(self.gp.Y)
            imp = mu - f_plus - self.xsi
        else:
            f_plus = np.amax(self.gp.Y)
            imp = f_plus + mu - self.xsi

        Z = np.empty_like(sigma)
        for i in range(len(sigma)):
            if sigma[i] > 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0
        ei = np.empty_like(sigma)
        for i in range(len(sigma)):
            if sigma[i] > 0:
                ei[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                ei[i] = 0

        return self.X_s[np.argmax(ei, 0)], ei
