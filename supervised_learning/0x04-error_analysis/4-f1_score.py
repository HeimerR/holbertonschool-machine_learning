#!/usr/bin/env python3
""" F1 score """
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """  calculates the F1 score of a confusion matrix """
    p = precision(confusion)
    s = sensitivity(confusion)
    return 2*p*s/(p+s)
