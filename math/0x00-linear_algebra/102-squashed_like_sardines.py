#!/usr/bin/env python3
""" Squashed Like Sardines """


def inception(mat1, mat2, axis, level):
    """ traverse """
    cat = []
    if level == axis:
        cat = mat1 + mat2
        return cat
    for i in range(len(mat1)):
        if type(mat1[i]) == list:
            cat.append(inception(mat1[i], mat2[i], axis, level + 1))
        else:
            cat.append(mat1 + mat2)
    return cat


def shape(matrix):
    """ matrix shape """
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape


def cat_matrices(mat1, mat2, axis=0):
    """ concatenates two matrices along a specific axis """
    shape1 = shape(mat1)
    shape2 = shape(mat2)
    if len(shape1) != len(shape2):
        return None
    level = 0
    cat = inception(mat1, mat2, axis, level)
    return cat
