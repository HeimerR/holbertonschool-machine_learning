#!/usr/bin/env python3
""" Definiteness """
import numpy as np


def definiteness(matrix):
    """ calculates the definiteness of a matrix:

    matrix is a numpy.ndarray of shape (n, n) whose definiteness
    should be calculated

    If matrix is not a numpy.ndarray, raise a TypeError with the message:
        "matrix must be a numpy.ndarray"

    If matrix is not a valid matrix, return None

    Return: the string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite if the matrix
        is positive definite, positive semi-definite, negative semi-definite,
        negative definite of indefinite, respectively

    If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # create a list of sub matrices (up-left to down-right)

    sub_m = [matrix[:i, :i] for i in range(1, matrix.shape[0]+1)]

    # calculate determinants for each sub matrix

    dets = np.array([np.linalg.det(a) for a in sub_m])

    # classify

    if len(matrix) == 1 and matrix[0][0] == 0:
        return None
    if all(dets > 0):
        return "Positive definite"
    if all(dets[::2] < 0) and all(dets[1::2] > 0):
        return "Negative definite"
    if dets[-1] != 0:
        return "Indefinite"
    if dets[-1] == 0 and all(dets[:-1] > 0):
        return "Positive semi-definite"
    if dets[-1] == 0 and all(dets[2:-1:2] < 0) and all(dets[1:-1:2] > 0):
        return "Negative semi-definite"
    return None
