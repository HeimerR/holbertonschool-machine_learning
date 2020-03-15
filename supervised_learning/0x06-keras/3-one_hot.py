#!/usr/bin/env python3
""" One Hot """
import numpy as np


def one_hot(labels, classes=None):
    """  converts a label vector into a one-hot matrix:

    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    if classes is None:
        classes = max(labels) - min(labels) + 1
    one_hot = np.zeros((len(labels), classes))
    axis = np.arange(len(labels))
    one_hot[axis, labels] = 1
    return one_hot
