#!/usr/bin/env python3
""" Dense Block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
        @X: output from the previous layer
        @nb_filters: integer representing the number of filters in X
        @growth_rate: growth rate for the dense block
        @layers: number of layers in the dense block

        bottleneck layers used for DenseNet-B

        All weights uses he normal initialization
        All convolutions are preceded by Batch Normalization and a
            rectified linear activation (ReLU), respectively

        Returns: The concatenated output of each layer within the Dense
            Block and the number of filters within the concatenated outputs.
    """

    init = K.initializers.he_normal(seed=None)
    for i in range(layers):
        norm_1 = K.layers.BatchNormalization()(X)
        active_1 = K.layers.Activation('relu')(norm_1)

        out_3 = K.layers.Conv2D(filters=4*growth_rate,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=init)(active_1)

        norm_2 = K.layers.BatchNormalization()(out_3)
        active_2 = K.layers.Activation('relu')(norm_2)

        out_4 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=init)(active_2)
        X = K.layers.concatenate([X, out_4])
        nb_filters += growth_rate

    return X, nb_filters
