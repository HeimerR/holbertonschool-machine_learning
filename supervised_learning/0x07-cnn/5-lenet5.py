#!/usr/bin/env python3
""" LeNet-5 (Keras) """
import tensorflow.keras as K


def lenet5(X):
    """ builds a modified version of the LeNet-5 architecture using keras:

        @X is a K.Input of shape (m, 28, 28, 1) containing the input images
            @m is the number of images

        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with
                valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
            All layers requiring initialization initializes their kernels with
                the he_normal initialization method
            All hidden layers requiring activation should use the
                relu activation function
        Returns: a K.Model compiled to use Adam optimization
            (with default hyperparameters) and accuracy metrics
    """
    model = K.Sequential()
    init = K.initializers.he_normal(seed=None)
    model.add(K.layers.Conv2D(filters=6,
                              kernel_size=5,
                              padding='same',
                              kernel_initializer=init,
                              activation='relu',
                              input_shape=(28, 28, 1)))

    model.add(K.layers.MaxPool2D(strides=2))

    model.add(K.layers.Conv2D(filters=16,
                              kernel_size=5,
                              padding='valid',
                              kernel_initializer=init,
                              activation='relu'))

    model.add(K.layers.MaxPool2D(strides=2))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(units=120,
                             kernel_initializer=init,
                             activation='relu'))

    model.add(K.layers.Dense(units=84,
                             kernel_initializer=init,
                             activation='relu'))

    model.add(K.layers.Dense(units=10,
                             kernel_initializer=init,
                             activation='softmax'))

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model