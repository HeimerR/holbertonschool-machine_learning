#!/usr/bin/env python3
""" Transfer Knowledge
    trains a convolutional neural network to classify the CIFAR 10 dataset

    Trained model saved in the current working directory as cifar10.h5
    Model saved is compiled
    Model saved has a validation accuracy of 88% or higher
    Script doesn't run when the file is imported
"""
import tensorflow.keras as K
import numpy as np


if __name__ == '__main__':
    """ trains the model and save it """
    input_tensor = K.Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)
    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    x_train = np.concatenate((x_train, np.flip(x_train, 2)), 0)
    y_train = np.concatenate((y_train, y_train), 0)

    model = K.applications.densenet.DenseNet121(include_top=False,
                                                pooling='max',
                                                input_tensor=input_tensor,
                                                weights='imagenet')

    output = model.layers[-1].output
    output = K.layers.Dense(512, activation='relu')(output)
    output = K.layers.Dropout(0.3)(output)
    output = K.layers.Dense(128, activation='relu')(output)
    output = K.layers.Dropout(0.3)(output)
    output = K.layers.Dense(10, activation='softmax')(output)

    model = K.models.Model(inputs=model.inputs, outputs=output)

    lrr = K.callbacks.ReduceLROnPlateau(
                                       monitor='val_acc',
                                       factor=.01,
                                       patience=3,
                                       min_lr=1e-5)

    es = K.callbacks.EarlyStopping(monitor='val_acc',
                                   mode='max',
                                   verbose=1,
                                   patience=10)

    mc = K.callbacks.ModelCheckpoint('cifar10.h5',
                                     monitor='val_acc',
                                     mode='max',
                                     verbose=1,
                                     save_best_only=True)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        callbacks=[es, mc, lrr],
                        epochs=30,
                        verbose=1)

    model.save('cifar10.h5')


def preprocess_data(X, Y):
    """ pre-processes the data for your model

        @X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
            where m is the number of data points
        @Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

        Returns: X_p, Y_p
            X_p: numpy.ndarray containing the preprocessed X
            Y_p: numpy.ndarray containing the preprocessed Y
    """

    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y
