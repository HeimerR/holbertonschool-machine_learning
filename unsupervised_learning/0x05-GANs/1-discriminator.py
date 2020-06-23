#!/usr/bin/env python3
""" Discriminator """
import numpy as np
import tensorflow as tf


def discriminator():
    """ creates a discriminator network for MNIST digits:

        - X is a tf.tensor containing the input to the discriminator network

        The network should have two layers:

            the first layer should have 128 nodes and use relu activation
            with name layer_1
            the second layer should have 1 node and use a sigmoid activation
            with name layer_2

        All variables in the network should have the scope discriminator
        with reuse=tf.AUTO_REUSE

        Returns Y, a tf.tensor containing the classification made
        by the discriminator
    """
    with tf.variable_scope('dis', reuse = reuse=tf.AUTO_REUSE):
        layer_1 = tf.layers.Dense(units = 128, name='layer_1'
                                  activation = tf.nn.relu)(X)

        layer_2 = tf.layers.Dense(units = 1, name='layer_2',
                                  activation=tf.nn.sigmoid)

        Y = layer_2(layer_1)

        return Y
