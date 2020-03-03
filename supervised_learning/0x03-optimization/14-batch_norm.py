#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network
    in tensorflow
    """
    x = tf.layers.Dense(units=n, activation=None)
    f = tf.layers.batch_normalization(inputs=x(prev))
    return activation(f)
