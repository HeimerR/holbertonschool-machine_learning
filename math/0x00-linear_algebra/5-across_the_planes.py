#!/usr/bin/env python3
""" adds two matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ adds two matrices element-wise """
    add = []
    if len(mat1) == len(mat2):
        for i in range(len(mat1)):
            if len(mat1[i]) == len(mat2[i]):
                add1 = [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
                add.append(add1)
            else:
                return None
        return add
    return None
