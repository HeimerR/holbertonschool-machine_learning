#!/usr/bin/env python3
""" DenseNet-121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds the DenseNet-121 architecture
        @growth_rate: growth rate
        @compression: compression factor

        Input data will have shape (224, 224, 3)
        All convolutions are preceded by Batch Normalization
            and a rectified linear activation (ReLU), respectively
        All weights uses he normal initialization
        Returns: the keras model
    """
    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]

    norm_1 = K.layers.BatchNormalization()(X)
    active_1 = K.layers.Activation('relu')(norm_1)

    out_1 = K.layers.Conv2D(filters=2*growth_rate,
                            kernel_size=7,
                            padding='same',
                            strides=2,
                            kernel_initializer=init)(active_1)

    id_1 = K.layers.MaxPool2D(pool_size=3,
                              strides=2,
                              padding='same')(out_1)

    d_1, nb_filters = dense_block(id_1, growth_rate*2, growth_rate, 6)

    for layer in layers:
        t_1, nb_filters = transition_layer(d_1, nb_filters, compression)
        d_1, nb_filters = dense_block(t_1, nb_filters, growth_rate, layer)

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         strides=None,
                                         padding='same')(d_1)

    out = K.layers.Dense(1000,
                         activation='softmax',
                         kernel_regularizer=K.regularizers.l2())(avg_pool)

    model = K.models.Model(inputs=X, outputs=out)

    return model
