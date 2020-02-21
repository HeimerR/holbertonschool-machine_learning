#!/usr/bin/env python3
""" Class NeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.forward_prop(X)
        return np.where(self.__A2 >= 0.5, 1, 0), self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural net """
        dz2 = A2 - Y
        m = A1.shape[1]
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1a = np.matmul(self.__W2.T, dz2)
        dz1b = A1 * (1 - A1)
        dz1 = dz1a * dz1b
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - (alpha * dw2).T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the neural network """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps = []
        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph is True:
            plt.plot(np.array(steps), np.array(costs))
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
