#!/usr/bin/env python3
""" function """


def matrix_shape(matrix):
    """ matrix shape """
    shape = [len(matrix)]
    while type(matrix[0]) == list:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape
