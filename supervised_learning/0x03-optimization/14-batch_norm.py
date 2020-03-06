#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network
    in tensorflow
    """
    gamma = tf.get_variable("gamma", [n])
    beta = tf.get_variable("beta", [n])
    x = tf.layers.Dense(units=n, activation=None)
    m, s = tf.nn.moments(x(prev), axes=[0], keep_dims=True)
    f = (tf.nn.batch_normalization(x(prev), mean=m, variance=s,
         offset=beta, scale=gamma, variance_epsilon=1e-8))
    return activation(f)
