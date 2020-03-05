#!/usr/bin/env python3
""" Sensitivity """
import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix """
    diagonal_tp = np.diagonal(confusion)
    tp_fn = np.sum(confusion, axis=1)
    return diagonal_tp / tp_fn
