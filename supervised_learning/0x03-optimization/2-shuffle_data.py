#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """  shuffles the data points in two matrices the same way """
    Xs = np.random.permutation(X)
    Ys = np.random.permutation(Y)
    return Xs, Ys
