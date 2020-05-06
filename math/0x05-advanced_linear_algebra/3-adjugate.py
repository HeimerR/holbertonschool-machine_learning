#!/usr/bin/env python3
""" Adjugate """


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

    # for general case --recursive

    det = 0
    for i, k in enumerate(matrix[0]):
        new_m = [[m[n] for n in range(len(m)) if n != i]for m in matrix[1:]]
        det += k * (-1)**(i) * determinant(new_m)

    return det


def adjugate(matrix):
    """ calculates the adjugate matrix of a matrix:

        matrix is a list of lists whose minor matrix should be calculated
        If matrix is not a list of lists, raise a TypeError with the message:
        "matrix must be a list of lists"

        If matrix is not square or is empty, raise a ValueError with
        the message: "matrix must be a non-empty square matrix"

        Returns: the adjugate matrix of matrix
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

            # set sub matrix -- colums-rows swap

            temp_m = [[matrix[n][m] for n in range(lm) if (n != j and m != i)]
                      for m in range(lm)]

            # delete empty nodes

            temp_m = [m for m in temp_m if len(m) == lm-1]

            # calculate and add determinant , multiplied by factor

            sub_new_matrix.append((-1)**(i+j)*determinant(temp_m))

        new_matrix.append(sub_new_matrix)

    return new_matrix
