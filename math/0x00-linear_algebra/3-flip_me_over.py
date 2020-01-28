#!/usr/bin/env python3
""" flip me over """


def matrix_transpose(matrix):
    """ transpose matrix """
    transpose = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return transpose
