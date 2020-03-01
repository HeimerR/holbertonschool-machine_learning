#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """  shuffles the data points in two matrices the same way """
    np.random.seed(0)
    X_shuffled = np.random.permutation(X)
    np.random.seed(0)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
