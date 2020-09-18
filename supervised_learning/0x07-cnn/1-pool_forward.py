#!/usr/bin/env python3
""" Inception Network with Keras """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network
        - You can assume the input data will have shape (224, 224, 3)
        - All convolutions inside and outside the inception block
          should use a rectified linear activation (ReLU)
        - You may use:
        inception_block = __import__('0-inception_block').inception_block
        Returns: the keras model
    """
    # Input Layer
    X = K.Input(shape=(224, 224, 3))
    # Convolutional layer
    conv = K.layers.Conv2D(64, (7, 7), padding='same',
                           activation='relu',
                           strides=2)(X)
    # Maxpooling layer
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(conv)
    # Convolutional layer
    conv = K.layers.Conv2D(64, (1, 1), padding='same',
                           activation='relu',
                           strides=1)(max_pool)
    # Convolutional layer
    conv = K.layers.Conv2D(192, (3, 3), padding='same',
                           activation='relu',
                           strides=1)(conv)
    # Maxpooling layer
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(conv)
    # Inception block
    inception = inception_block(max_pool, [64, 96, 128, 16, 32, 32])
    # Inception block
    inception = inception_block(inception, [128, 128, 192, 32, 96, 64])
    # Maxpooling layer
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(inception)
    # Inception block
    inception = inception_block(max_pool, [192, 96, 208, 16, 48, 64])
    # Inception block
    inception = inception_block(inception, [160, 112, 224, 24, 64, 64])
    # Inception block
    inception = inception_block(inception, [128, 128, 256, 24, 64, 64])
    # Inception block
    inception = inception_block(inception, [112, 144, 288, 32, 64, 64])
    # Inception block
    inception = inception_block(inception, [256, 160, 320, 32, 128, 128])
    # Maxpooling layer
    max_pool = K.layers.MaxPooling2D((3, 3), strides=2,
                                     padding='same')(inception)
    # Inception block
    inception = inception_block(max_pool, [256, 160, 320, 32, 128, 128])
    # Inception block
    inception = inception_block(inception, [384, 192, 384, 48, 128, 128])
    # Average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=1,
                                         padding="same")(inception)
    # Dropout layer
    drop = K.layers.Dropout(0.4)(avg_pool)
    # Linear activation
    linear = K.activations.linear(drop)
    # Softmax dense layer
    softmax = K.layers.Dense(1000, activation='softmax')(linear)
    # Network model
    network = K.models.Model(inputs=X, outputs=softmax)

    return network
