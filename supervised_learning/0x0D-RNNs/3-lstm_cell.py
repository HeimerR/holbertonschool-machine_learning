#!/usr/bin/env python3
""" LSTM Cell """
import numpy as np


class LSTMCell:
    """ represents an LSTM unit """
    def __init__(self, i, h, o):
        """
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs

            Creates the public instance attributes
                Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
                that represent the weights and biases of the cell

                Wf and bf are for the forget gate
                Wu and bu are for the update gate
                Wc and bc are for the intermediate cell state
                Wo and bo are for the output gate
                Wy and by are for the outputs
            The weights should be initialized using a random
            normal distribution in the order listed above
            The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(h+i, h))
        self.Wu = np.random.normal(size=(h+i, h))
        self.Wc = np.random.normal(size=(h+i, h))
        self.Wo = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """ sigmoid function """
        return 1/(1 + np.exp(-x))

    def softmax2(self, x):
        """ softmax function """
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step
            - x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
                - m is the batche size for the data
                - h_prev is a numpy.ndarray of shape (m, h) containing
                the previous hidden state
            - c_prev is a numpy.ndarray of shape (m, h) containing the
                previous cell state

        Returns: h_next, c_next, y
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """
        matrix = np.concatenate((h_prev, x_t), axis=1)
        u_gate = self.sigmoid(np.matmul(matrix, self.Wu) + self.bu)
        f_gate = self.sigmoid(np.matmul(matrix, self.Wf) + self.bf)
        o_gate = self.sigmoid(np.matmul(matrix, self.Wo) + self.bo)
        c_tilde = np.tanh(np.matmul(matrix, self.Wc) + self.bc)
        c_next = u_gate * c_tilde + f_gate * c_prev
        h_next = o_gate * np.tanh(c_next)

        # you can call softmax instead softmax2 - same op
        y = self.softmax2(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
