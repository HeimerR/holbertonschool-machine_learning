#!/usr/bin/env python3
""" The Whole Barn """


def shape(matrix):
    """ matrix shape """
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def add_matrices(mat1, mat2):
    """  adds two matrices """
    if shape(mat1) != shape(mat2):
        return None
    dimens = 0
    add = []
    temp = mat1
    while type(temp[0]) == list:
        dimens += 1
        temp = temp[0]
    print(dimens)
    return None
