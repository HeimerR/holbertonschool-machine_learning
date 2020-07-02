#!/usr/bin/env python3
""" RNN Cell """
import numpy as np


class RNNCell:
    """  cell of a simple RNN """
    def __init__(self, i, h, o):
        """ - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs

            Wh, Wy, bh, by that represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
            The weights are initialized using a random normal distribution
                in the order listed above
            The weights will be used on the right side for matrix
                multiplication
            The biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax2(self, x):
        """ softmax function """
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """  performs forward propagation for one time step
            - x_t is a numpy.ndarray of shape (m, i) that contains the
                data input for the cell
            - m is the batche size for the data
            - h_prev is a numpy.ndarray of shape (m, h) containing the
                previous hidden state

        Returns: h_next, y
            - h_next is the next hidden state
            - y is the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax2(y)  # you can call softmax instead softmax2

        return h_next, y
