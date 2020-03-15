#!/usr/bin/env python3
"""Macsdascdacascdasdcasdc"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """sdacascdasdcadscasdcc"""
    r = K.regularizers.l2
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             input_shape=(nx,),
                             kernel_regularizer=r(lambtha)))
    for ly, a in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(ly, activation=a,
                  kernel_regularizer=r(lambtha)))
    return model
