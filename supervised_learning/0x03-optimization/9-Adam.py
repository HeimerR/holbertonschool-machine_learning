#!/usr/bin/env python3
""" Adam """


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ updates a variable in place using the Adam optimization algorithm """
    Vdv = beta1*v + (1-beta1)*grad
    Sdv = beta2*s + (1-beta2)*(grad**2)

    Vdv_corrected = Vdv / (1-beta1**t)
    Sdv_corrected = Sdv / (1-beta2**t)
    var_updated = var - alpha*(Vdv_corrected/((Sdv_corrected**(1/2))+epsilon))
    return var_updated, Vdv, Sdv
