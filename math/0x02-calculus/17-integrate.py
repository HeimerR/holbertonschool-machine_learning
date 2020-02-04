#!/usr/bin/env python3
""" integrate """


def poly_integral(poly, C=0):
    """  calculates the integral of a polynomial """
    if not isinstance(C, (int, float)) or not isinstance(poly, list):
        return None
    if not all(isinstance(n, (int, float)) for n in poly):
        return None
    i = len(poly) - 1
    while poly[i] == 0:
        poly.pop(i)
        i -= 1
    integral = [float(C)] + [poly[i] / (i + 1) for i in range(len(poly))]
    return [int(coef) if coef.is_integer() else coef for coef in integral]
