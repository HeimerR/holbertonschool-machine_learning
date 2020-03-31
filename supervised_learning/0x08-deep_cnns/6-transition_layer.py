#!/usr/bin/env python3
""" Transition Layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer

        @X: output from the previous layer
        @nb_filters: integer representing the number of filters in X
        @compression: compression factor for the transition layer

        implemented compression as used in DenseNet-C
        All weights uses he normal initialization
        All convolutions are preceded by Batch Normalization and a rectified
            linear activation (ReLU), respectively
        Returns: The output of the transition layer and the number of filters
        within the output, respectively
    """
    F = int(nb_filters*compression)

    init = K.initializers.he_normal(seed=None)
    norm_1 = K.layers.BatchNormalization()(X)
    active_1 = K.layers.Activation('relu')(norm_1)
    out_1 = K.layers.Conv2D(filters=F,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init)(active_1)

    avg_pool = K.layers.AveragePooling2D(pool_size=2,
                                         strides=None,
                                         padding='same')(out_1)

    return avg_pool, F
