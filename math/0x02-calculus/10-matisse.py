#!/usr/bin/env python3
""" Derivative """


def poly_derivative(poly):
    """  calculates the derivative of a polynomial """
    if type(poly) != list:
        return None
    if not all(isinstance(n, (int, float)) for n in poly):
        return None
    dx = [i*poly[i] for i in range(len(poly))]
    if len(set(dx)) == 1:
        return [0]
    return dx[1:]
