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
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for ly in range(self.__L):
            Zp = np.matmul(self.__weights["W"+str(ly+1)],
                           self.__cache["A"+str(ly)])
            Z = Zp + self.__weights["b"+str(ly+1)]
            if ly == self.__L - 1:
                t = np.exp(Z)
                self.__cache["A"+str(ly+1)] = (t/np.sum(t, axis=0,
                                               keepdims=True))
            else:
                if self.__activation == 'sig':
                    self.__cache["A"+str(ly+1)] = 1/(1+np.exp(-Z))
                else:
                    self.__cache["A"+str(ly+1)] = np.tanh(Z)

        return self.__cache["A"+str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression
            @Y: one-hot numpy.ndarray of shape (classes, m)
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.forward_prop(X)
        tmp = np.amax(self.__cache["A"+str(self.__L)], axis=0)
        return (np.where(self.__cache["A"+str(self.__L)] == tmp, 1, 0),
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
                if self.__activation == 'sig':
                    d2 = (self.__cache["A"+str(ly+1)] *
                          (1-self.__cache["A"+str(ly+1)]))
                else:
                    d2 = 1-self.__cache["A"+str(ly+1)]**2
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network
            @X: Input data
            @Y: Correct labels for the input data
            @iterations: Number of iterations to train over
            @alpha: Learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        if verbose is True or graph is True:
            if step > iterations:
                raise ValueError("step must be positive and <= iterations")
        it_x = []
        cost_y = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            if (i % step) == 0:
                a = 'A' + str(self.__L)
                cost = self.cost(Y, self.__cache[a])
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    it_x.append(i)
                    cost_y.append(cost)
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        if graph is True:
            plt.plot(it_x, cost_y)
            plt.title("Training Cost")
            plt.xlabel("iterations")
            plt.ylabel("cost")
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
