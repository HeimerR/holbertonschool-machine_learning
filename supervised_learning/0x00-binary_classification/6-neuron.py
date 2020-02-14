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
        C = np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return (-1/(Y.shape[1])) * C

    def evaluate(self, X, Y):
        """ evaluate output """
        self.forward_prop(X)
        return np.where(self.__A >= 0.5, 1, 0), self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        dz = A - Y
        m = Y.shape[1]
        dw = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            print(i)
            self.evaluate(X, Y)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
