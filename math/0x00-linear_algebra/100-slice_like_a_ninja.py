#!/usr/bin/env python3
""" Slice Like A Ninja """


def np_slice(matrix, axes={}):
    """ slices a matrix along a specific axes """
    slice_all = slice(None, None, None)
    shape_matrix = matrix.shape
    slice_list = [slice_all] * len(shape_matrix)
    for key, value in sorted(axes.items()):
        slice_object = slice(*value)
        slice_list[key] = slice_object
    matrix = matrix[tuple(slice_list)]
    return matrix
