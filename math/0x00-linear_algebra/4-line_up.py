#!/usr/bin/env python3
""" Line UP """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    if len(arr1) == len(arr2):
        add = [arr1[i] + arr2[i] for i in range(len(arr1))]
        return add
    return None
