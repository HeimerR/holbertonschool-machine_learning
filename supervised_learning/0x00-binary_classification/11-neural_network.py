#!/usr/bin/env python3
""" Class NeuralNetwork """
import numpy as np


class NeuralNetwork:
    """ class NeuralNetwork """
    def __init__(self, nx, nodes):
        """ init for NeuralNetwork """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter for W1 """
        return self.__W1

    @property
    def W2(self):
        """ getter for W2 """
        return self.__W2

    @property
    def b1(self):
        """ getter for b1 """
        return self.__b1

    @property
    def b2(self):
        """ getter for b2 """
        return self.__b2

    @property
    def A1(self):
        """ getter for A1 """
        return self.__A1

    @property
    def A2(self):
        """ getter for A12 """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        C = np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return (-1/(Y.shape[1])) * C
