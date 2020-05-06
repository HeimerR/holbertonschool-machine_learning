#!/usr/bin/env python3
""" Minor """


def determinant(matrix):
    """ matrix is a list of lists whose determinant should be calculated
        If matrix is not a list of lists, raise a TypeError with the message:
        "matrix must be a list of lists"
        If matrix is not square, raise a ValueError with the message:
        "matrix must be a square matrix"
        The list [[]] represents a 0x0 matrix
        Returns: the determinant of matrix
    """
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    lm = len(matrix)
    if lm == 1 and len(matrix[0]) == 0:
        return 1
    if not all([len(n) == lm for n in matrix]):
        raise ValueError("matrix must be a square matrix")
    if lm == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    # for 2x2 matrix minimun case
    if lm == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = a*d - c*b
        return det
    # for general case
    det = 0
    for i, k in enumerate(matrix[0]):
        new_matrix = []
        for m in matrix[1:]:
            sub_matrix = []
            for n in range(len(m)):
                if n != i:
                    sub_matrix.append(m[n])
            new_matrix.append(sub_matrix)

        det += k * (-1)**(i) * determinant(new_matrix)

    return det


def cofactor(matrix):
    """ calculates the minor matrix of a matrix:

        matrix is a list of lists whose minor matrix should be calculated
        If matrix is not a list of lists, raise a TypeError with the message:
        "matrix must be a list of lists"

        If matrix is not square or is empty, raise a ValueError with
        the message: "matrix must be a non-empty square matrix"

        Returns: the minor matrix of matrix
    """
    if (type(matrix) != list or len(matrix) == 0 or
       not all([type(m) == list for m in matrix])):
        raise TypeError("matrix must be a list of lists")
    lm = len(matrix)
    if lm == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all([len(n) == lm for n in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if lm == 1 and len(matrix[0]) == 1:
        return [[1]]
    new_matrix = []
    for i in range(lm):
        sub_new_matrix = []
        for j in range(lm):

            temp_m = []
            for m in range(lm):
                sub_m = []
                for n in range(lm):
                    if n != j and m != i:
                        sub_m.append(matrix[m][n])
                if len(sub_m) == lm - 1:
                    temp_m.append(sub_m)

            sub_new_matrix.append((-1)**(i+j)*determinant(temp_m))
        new_matrix.append(sub_new_matrix)
    return new_matrix
