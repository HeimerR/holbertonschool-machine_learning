#!/usr/bin/env python3
""" Identity Block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """  builds an identity block

        @A_prev: output from the previous layer
        @filters tuple or list containing F11, F3, F12, respectively:
            @F11 number of filters in the first 1x1 convolution
            @F3 number of filters in the 3x3 convolution
            @F12 number of filters in the second 1x1 convolution
        All convolutions inside the block are followed by batch normalization
        along the channels axis and a rectified linear activation (ReLU).
        All weights uses he normal initialization
        Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    out_1 = K.layers.Conv2D(filters=F11,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init)(A_prev)

    norm_1 = K.layers.BatchNormalization()(out_1)
    active_1 = K.layers.Activation('relu')(norm_1)

    out_3 = K.layers.Conv2D(filters=F3,
                            kernel_size=3,
                            padding='same',
                            kernel_initializer=init)(active_1)

    norm_3 = K.layers.BatchNormalization()(out_3)
    active_3 = K.layers.Activation('relu')(norm_3)

    out_12 = K.layers.Conv2D(filters=F12,
                             kernel_size=1,
                             padding='same',
                             kernel_initializer=init)(active_3)

    norm_12 = K.layers.BatchNormalization()(out_12)

    adds = K.layers.Add()([norm_12, A_prev])
    out = K.layers.Activation('relu')(adds)

    return out
