#!/usr/bin/env python3
""" Class Neuron """
import numpy as np


class Neuron:
    """ class neuron """
    def __init__(self, nx):
        """ init for  neuron """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.normal(size=self.nx).reshape(nx, 1).T
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ b getter """
        return self.__b

    @property
    def A(self):
        """ A getter """
        return self.__A

    def forward_prop(self, X):
        """ forward propagation """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ cost function """
        C = np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        return -C/Y.shape[1]

    def evaluate(self, X, Y):
        """ evaluate output """
        self.forward_prop(X)
        return np.where(self.__A >= 0.5, 1, 0), self.cost(Y, self.__A)
