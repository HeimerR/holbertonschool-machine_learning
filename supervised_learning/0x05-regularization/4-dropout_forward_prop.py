#!/usr/bin/env python3
""" Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """  conducts forward propagation using Dropout """
    cache = {}
    cache["A0"] = X
    for ly in range(L):
        Zp = np.matmul(weights["W"+str(ly+1)], cache["A"+str(ly)])
        Z = Zp + weights["b"+str(ly+1)]
        if ly == L - 1:
            t = np.exp(Z)
            cache["A"+str(ly+1)] = (t/np.sum(t, axis=0,
                                           keepdims=True))
        else:
            cache["A"+str(ly+1)] = np.tanh(Z)
        drop = np.random.binomial(len(weights["W"+str(ly+1)]), keep_prob, size=cache["A"+str(ly+1)].shape)
        cache["A"+str(ly+1)] *= drop
        if ly != 0 and ly != L-1:
            cache["D"+str(ly)] = drop
    return cache
