#!/usr/bin/env python3
""" Transfer Knowledge
    trains a convolutional neural network to classify the CIFAR 10 dataset

    Trained model saved in the current working directory as cifar10.h5
    Model saved is compiled
    Model saved has a validation accuracy of 88% or higher
    Script doesn't run when the file is imported
"""
import tensorflow.keras as K


if __name__ == '__main__':
    input_tensor = K.Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = K.applications.densenet.preprocess_input(x_train)
    y_train = K.utils.to_categorical(y_train, 10)
    x_test = K.applications.densenet.preprocess_input(x_test)
    y_test = K.utils.to_categorical(y_test, 10)

    model = K.applications.densenet.DenseNet121(include_top=False, pooling='avg', input_tensor=input_tensor, weights='imagenet')
    for layer in model.layers:
	    layer.trainable = False
    output = model.layers[-1].output
    #output = K.layers.Flatten()(output)
    output = K.layers.Dense(512, activation='relu')(output)
    output = K.layers.Dense(10, activation='softmax')(output)
    model = K.models.Model(inputs=model.inputs, outputs=output)
    model.summary()
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])
    history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=64,
                    epochs=5,
                    verbose=1)
    """
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)
    """
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
