#!/usr/bin/env python3
""" L2 Regularization Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization """
    Frobenius = 0
    for v in weights.values():
        Frobenius += np.linalg.norm(v)
    return cost + (lambtha/(2*m)) * Frobenius
