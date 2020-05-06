#!/usr/bin/env python3
""" #!/usr/bin/env python3 """


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
