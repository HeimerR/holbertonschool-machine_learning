#!/usr/bin/env python3
""" The Whole Barn """
import numpy as np


def add_matrices(mat1, mat2):
    """  adds two matrices """
    m1 = np.asarray(mat1)
    m2 = np.asarray(mat2)
    if m1.shape != m2.shape:
        return None
    add = m1 + m2
    return add.tolist()
