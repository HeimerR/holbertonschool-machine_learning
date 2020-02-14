#!/usr/bin/env python3
""" Class Neuron """
import numpy as np


class Neuron:
    """ class neuron """
    def __init__(self, nx):
        """ init fot neuron """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.normal(size=nx).reshape(nx, 1).T
        self.b = 0
        self.A = 0
