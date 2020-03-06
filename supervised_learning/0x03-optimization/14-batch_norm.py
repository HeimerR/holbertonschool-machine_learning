#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates a batch normalization layer for a neural network
    in tensorflow
    """
    beta = (tf.get_variable("beta", [n],
            initializer=tf.zeros_initializer(), trainable=True))
    gamma = (tf.get_variable("gamma", [n],
             initializer=tf.ones_initializer(), trainable=True))

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)

    m, s = tf.nn.moments(x(prev), axes=[0])

    f = (tf.nn.batch_normalization(x(prev), mean=m, variance=s,
         offset=beta, scale=gamma, variance_epsilon=1e-8))

    return activation(f)
