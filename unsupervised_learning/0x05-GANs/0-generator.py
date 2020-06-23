#!/usr/bin/env python3
""" Generator """
import numpy as np
import tensorflow as tf


def generator(Z):
    """ creates a simple generator network for MNIST digits
        - Z is a tf.tensor containing the input to the generator network
        The network should have two layers:
            the first layer should have 128 nodes and use relu
            activation with name layer_1

            the second layer should have 784 nodes and use a
            sigmoid activation with name layer_2

        Returns X, a tf.tensor containing the generated image
    """
    layer_1 = tf.layers.Dense(units = 128, name='layer_1',
	                      activation = tf.nn.relu)(Z)

    layer_2 = tf.layers.Dense(units = 784, name='layer_2',
                              activation = tf.nn.sigmoid)

    X = layer_2(layer_1)

    return X
