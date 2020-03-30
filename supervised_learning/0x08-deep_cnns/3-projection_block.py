#!/usr/bin/env python3
""" Projection Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block
    @A_prev: the output from the previous layer
    @filters: tuple or list containing F11, F3, F12, respectively:
        @F11: number of filters in the first 1x1 convolution
        @F3: number of filters in the 3x3 convolution
        @F12: number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
    @s is the stride of the first convolution in both the main
        path and the shortcut connection
    All convolutions inside the block are followed by batch normalization
        along the channels axis and a rectified linear activation (ReLU).
    All weights uses he normal initialization
    Returns: the activated output of the projection block
    """

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    out_1 = K.layers.Conv2D(filters=F11,
                            kernel_size=1,
                            padding='same',
                            strides=s,
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

    out_shorcut = K.layers.Conv2D(filters=F12,
                                  kernel_size=1,
                                  padding='same',
                                  strides=s,
                                  kernel_initializer=init)(A_prev)

    norm_shortcut = K.layers.BatchNormalization()(out_shorcut)

    adds = K.layers.Add()([norm_12, norm_shortcut])
    out = K.layers.Activation('relu')(adds)

    return out
