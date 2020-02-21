#!/usr/bin/env python3
""" DeepNeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ class DeepNeuralNetwork """
    def __init__(self, nx, layers):
        """ init DeepNeuralNetwork """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        if not all(n > 0 for n in layers):
            raise TypeError('layers must be a list of positive integers')
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for ly in range(self.__L):
            self.__weights["b"+str(ly+1)] = np.zeros((layers[ly], 1))
            if ly == 0:
                heetal = np.random.randn(layers[ly], nx) * np.sqrt(2/nx)
                self.__weights["W"+str(ly+1)] = heetal
            else:
                factor = np.sqrt(2/layers[ly-1])
                heetal = np.random.randn(layers[ly], layers[ly-1]) * factor
                self.__weights["W"+str(ly+1)] = heetal

    @property
    def L(self):
        """ The number of layers in the neural network """
        return self.__L

    @property
    def cache(self):
        """ hold all intermediary values of the network """
        return self.__cache

    @property
    def weights(self):
        """ hold all weights and biased of the network """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for ly in range(self.__L):
            Zp = np.matmul(self.__weights["W"+str(ly+1)],
                           self.__cache["A"+str(ly)])
            Z = Zp + self.__weights["b"+str(ly+1)]
            self.__cache["A"+str(ly+1)] = 1/(1+np.exp(-Z))

        return self.__cache["A"+str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        C = np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return (-1/(Y.shape[1])) * C

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.forward_prop(X)
        return (np.where(self.__cache["A"+str(self.__L)] >= 0.5, 1, 0),
                self.cost(Y, self.__cache["A"+str(self.__L)]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        tmp_W = self.__weights.copy()
        m = Y.shape[1]
        for ly in reversed(range(self.__L)):
            if ly == self.__L - 1:
                dz = self.__cache["A"+str(ly+1)] - Y
                dw = np.matmul(self.__cache["A"+str(ly)], dz.T) / m
            else:
                d1 = np.matmul(tmp_W["W"+str(ly+2)].T, dzp)
                d2 = (self.__cache["A"+str(ly+1)] *
                      (1-self.__cache["A"+str(ly+1)]))
                dz = d1 * d2
                dw = np.matmul(dz, self.__cache["A"+str(ly)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if ly == self.__L - 1:
                self.__weights["W"+str(ly+1)] = (tmp_W["W"+str(ly+1)] -
                                                 (alpha * dw).T)
            else:
                self.__weights["W"+str(ly+1)] = (tmp_W["W"+str(ly+1)] -
                                                 (alpha * dw))
            self.__weights["b"+str(ly+1)] = tmp_W["b"+str(ly+1)] - alpha * db
            dzp = dz

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the deep neural network """
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
            cost = self.cost(Y, self.__cache["A"+str(self.__L)])
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        plt.plot(np.array(steps), np.array(costs))
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.suptitle("Training Cost")
        plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            f = open(filename, 'rb')
            obj = pickle.load(f)
            f.close()
            return obj
        except FileNotFoundError:
            return None
