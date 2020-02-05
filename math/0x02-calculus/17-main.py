#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))
poly = [5, 3, 0, 1]
print(poly_integral(poly, 100))
poly = [5, 3, 0, 1]
print(poly_integral(poly, 5.2))
poly = [5.9, 3.0, 0, 1]
print(poly_integral(poly))
poly = [5, 3, 0, 1]
print(poly_integral(poly, 'p'))
poly = [5, 'o', 0, 1]
print(poly_integral(poly))
