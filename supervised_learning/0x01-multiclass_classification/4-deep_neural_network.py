#!/usr/bin/env python3
""" Deep neural network """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """ defines a deep neural network """
    def __init__(self, nx, layers, activation='sig'):
        """ Class constructor
            @nx: number of input features
            @layers: number of nodes in each layer of the network
            @L: number of layers in the neural network
            @cache: dictionary to hold all intermediary values of the network
            @weights: dictionary to hold all weights and biased of the network
            @activation: represents the type of activation function used in
                         the hidden layers
                sig represents a sigmoid activation
                tanh represents a tanh activation
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        if activation != "sig" and activation != "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            w = "W" + str(i + 1)
            b = "b" + str(i + 1)

            if i == 0:
                self.weights[w] = np.random.randn(layers[i], nx)\
                                  * np.sqrt(2. / nx)
            else:
                self.weights[w] = np.random.randn(layers[i], layers[i - 1])\
                                  * np.sqrt(2 / layers[i - 1])
            self.weights[b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function """
        return self.__L

    @property
    def cache(self):
        """ Getter function """
        return self.__cache

    @property
    def weights(self):
        """ Getter function """
        return self.__weights

    @property
    def activation(self):
        """ Getter function """
        return self.__activation

    def forward_prop(self, X):
        """ Activation function
            Sigmoid Forward propagation
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            a = 'A' + str(i + 1)
            Z = np.dot(self.__weights[w], self.__cache['A' + str(i)])\
                + self.__weights[b]
            if i + 1 == self.L:
                exp = np.exp(Z)
                self.__cache[a] = exp / np.sum(exp, axis=0, keepdims=True)
            elif self.__activation == 'sig':
                self.__cache[a] = 1 / (1 + np.exp(-Z))
            elif self.__activation == 'tanh':
                self.__cache[a] = np.tanh(Z)
        return self.__cache[a], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            @Y: one-hot numpy.ndarray of shape (classes, m)
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron
            @Y: one-hot numpy.ndarray of shape (classes, m)
        """
        self.forward_prop(X)
        i = 'A' + str(self.__L)
        A = self.__cache[i]
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Gradient descent
            @Y: Labels for the input data
            @cache: dictionary containing all the intermediary
                    values of the network
            @alpha: Learning rate
        """
        m = len(Y[0])
        if self.__activation == 'sig':
            dz = self.__cache['A' + str(self.__L)] - Y
        if self.__activation == 'tanh':
            dz = self.__cache['A' + str(self.__L)] - Y**2
        for i in range(self.__L, 0, -1):
            a = 'A' + str(i - 1)
            w = 'W' + str(i)
            b = 'b' + str(i)
            A = self.__cache[a]
            dw = (1 / m) * np.dot(dz, np.transpose(A))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            if self.__activation == 'sig':
                dz = np.dot(np.transpose(self.__weights[w]), dz) * A * (1 - A)
            if self.__activation == 'tanh':
                dz = np.dot(np.transpose(self.__weights[w]), dz) * (1 - A**2)
            self.__weights[w] = self.__weights[w] - (alpha * dw)
            self.__weights[b] = self.__weights[b] - (alpha * db)

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
        if graph is True:
            plt.plot(np.array(steps), np.array(costs))
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if len(filename.split('.')) == 1:
            filename += '.pkl'
        try:
            file_object = open(filename, 'wb')
            pickle.dump(self, file_object)
            file_object.close()
        except Exception as e:
            return None

    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            file_object = open(filename, 'rb')
            a = pickle.load(file_object)
            file_object.close()
            return a
        except Exception as e:
            return None
