#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ performs forward propagation for a simple RNN:

        - rnn_cell is an instance of RNNCell that will
            be used for the forward propagation
        - X is the data to be used, given as a numpy.ndarray
            of shape (t, m, i)
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        - h_0 is the initial hidden state, given as a numpy.ndarray
            of shape (m, h)
            - h is the dimensionality of the hidden state

        Returns: H, Y
            - H is a numpy.ndarray containing all of the hidden states
            - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.empty((t+1, m, h))
    H[0] = h_0
    for ts in range(0, t):
        H[ts+1], y = rnn_cell.forward(H[ts], X[ts])
        if ts == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    o = Y.shape[-1]
    return H, Y.reshape(t, m, o)
