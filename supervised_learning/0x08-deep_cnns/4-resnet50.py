#!/usr/bin/env python3
"""  ResNet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """  builds the ResNet-50 architecture

    Input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks are followed
    by batch normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.

    All weights uses he normal initialization
    Returns: the keras model
    """
    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    out_1 = K.layers.Conv2D(filters=64,
                            kernel_size=7,
                            padding='same',
                            strides=2,
                            kernel_initializer=init)(X)

    norm_1 = K.layers.BatchNormalization()(out_1)
    active_1 = K.layers.Activation('relu')(norm_1)

    id_1 = K.layers.MaxPool2D(pool_size=3,
                              strides=2,
                              padding='same')(active_1)

    id_1 = projection_block(id_1, [64, 64, 256], 1)

    for i in range(2):
        id_1 = identity_block(id_1, [64, 64, 256])

    id_2 = projection_block(id_1, [128, 128, 512])

    for i in range(3):
        id_2 = identity_block(id_2, [128, 128, 512])

    id_3 = projection_block(id_2, [256, 256, 1024])

    for i in range(5):
        id_3 = identity_block(id_3, [256, 256, 1024])

    id_4 = projection_block(id_3, [512, 512, 2048])

    for i in range(2):
        id_4 = identity_block(id_4, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         strides=None,
                                         padding='same')(id_4)
    out = K.layers.Dense(1000,
                         activation='softmax',
                         kernel_regularizer=K.regularizers.l2())(avg_pool)

    model = K.models.Model(inputs=X, outputs=out)

    return model
