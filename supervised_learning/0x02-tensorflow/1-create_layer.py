#!/usr/bin/env python3
""" Layers """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ creates layers """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor_l = (tf.layers.dense(prev, units=n, activation=activation,
                kernel_initializer=initialize))
    return tensor_l
