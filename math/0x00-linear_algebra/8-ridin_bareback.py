#!/usr/bin/env python3
""" Ridin’ Bareback """


def mat_mul(mat1, mat2):
    """  performs matrix multiplication """
    if len(mat1[0]) != len(mat2):
        return None
    mul = []
    for i in range(len(mat1)):
        mul_row = []
        for j in range(len(mat2[0])):
            item = 0
            for k in range(len(mat2)):
                item += mat1[i][k] * mat2[k][j]
            mul_row.append(item)
        mul.append(mul_row)
    return mul
