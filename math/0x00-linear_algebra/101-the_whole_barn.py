#!/usr/bin/env python3
""" The Whole Barn """


def shape(matrix):
    """ matrix shape """
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def inception(mat1, mat2):
    """ traverse """
    add = []
    for i in range(len(mat1)):
        if type(mat1[i]) == list:
            add.append(inception(mat1[i], mat2[i]))
        else:
            add.append(mat1[i] + mat2[i])
    return add


def add_matrices(mat1, mat2):
    """ adds two matrices """
    shape1 = shape(mat1)
    shape2 = shape(mat2)
    tempshape = shape1
    if shape1 != shape2:
        return None
    add = inception(mat1, mat2)
    return add
