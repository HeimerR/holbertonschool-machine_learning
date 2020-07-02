#!/usr/bin/env python3
""" GRU Cell  """
import numpy as np


class GRUCell:
    """ represents a gated recurrent unit """
    def __init__(self, i, h, o):
        """
            - i is the dimensionality of the data
            - h is the dimensionality of the hidden state
            - o is the dimensionality of the outputs

            Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh,
            by that represent the weights and biases of the cell

            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output

            The weights should be initialized using a random normal
            distribution
            in the order listed above
            The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(h+i, h))
        self.Wr = np.random.normal(size=(h+i, h))
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax2(self, x):
        """ softmax function """
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """ sigmoid function """
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        """ softmax function """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                - m is the batche size for the data
            - h_prev is a numpy.ndarray of shape (m, h) containing the
                previous hidden state

            The weights will be used on the right side for matrix
                multiplication
            Returns: h_next, y
                - h_next is the next hidden state
                - y is the output of the cell
        """
        matrix = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(matrix, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(matrix, self.Wr) + self.br)

        matrix2 = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(matrix2, self.Wh) + self.bh)
        h_next = z_gate * h_tilde + (1 - z_gate) * h_prev

        # you can call softmax instead softmax2 - same op
        y = self.softmax2(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
