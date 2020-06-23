#!/usr/bin/env python3
""" Train Discriminator """
import numpy as np
import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """ creates the loss tensor and training op for the discriminator:

        - Z is the tf.placeholder that is the input for the generator
        - X is the tf.placeholder that is the input for the discriminator
        The discriminator should minimize the negative minimax loss
        The discriminator should be trained using Adam optimization
        The generator should NOT be trained

        Returns: loss, train_op
            loss is the discriminator loss
            train_op is the training operation for the discriminator
    """

