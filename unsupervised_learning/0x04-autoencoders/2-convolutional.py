#!/usr/bin/env python3
""" Convolutional Autoencoder """
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """ Creates an convolutional autoencoder:

        - input_dims is a tuple of integers containing
                the dimensions of the model input
        - filters is a list containing the number of filters
                for each convolutional layer in the encoder, respectively
                the filters should be reversed for the decoder
        - latent_dims is a tuple of integers containing the dimensions
                of the latent space representation
        Each convolution in the encoder should use a kernel size of (3, 3)
        with same padding and relu activation, followed by max pooling of
        size (2, 2)
        Each convolution in the decoder, except for the last two, should use
        a filter size of (3, 3) with same padding and relu activation, followed
        by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have only 1 filter with sigmoid activation
        and no upsampling

        Returns: encoder, decoder, auto
                - encoder is the encoder model
                - decoder is the decoder model
                - auto is the full autoencoder model
        The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    """
    input_encoder = K.Input(shape=input_dims)

    output = K.layers.Conv2D(filters=filters[0],
                             kernel_size=3,
                             padding='same',
                             activation='relu')(input_encoder)
    output = K.layers.MaxPool2D(pool_size=(2, 2))(output)

    for i in range(1, len(filters)):
        output = K.layers.Conv2D(filters=filters[i],
                                 kernel_size=3,
                                 padding='same',
                                 activation='relu')(output)
        output = K.layers.MaxPool2D(pool_size=(2, 2))(output)

    out_encoder = K.layers.Reshape(latent_dims)(output)

    input_decoder = K.Input(shape=latent_dims)

    output2 = K.layers.Conv2D(filters=filters[-1],
                              kernel_size=3,
                              padding='same',
                              activation='relu')(input_decoder)
    output2 = K.layers.UpSampling2D(2)(output2)

    for i in range(len(filters)-2, 2, -1):
        print(i)
        output2 = K.layers.Conv2D(filters=filters[i],
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu')(output2)
        output2 = K.layers.UpSampling2D(2)(output2)

    output2 = K.layers.Conv2D(filters=filters[1],
                              kernel_size=3,
                              padding='same',
                              activation='relu')(output2)
    output2 = K.layers.UpSampling2D(2)(output2)

    out_decoder = K.layers.Conv2D(filters=1,
                                  kernel_size=3,
                                  padding='same',
                                  activation='sigmoid')(output2)

    encoder = K.models.Model(inputs=input_encoder, outputs=out_encoder)
    decoder = K.models.Model(inputs=input_decoder, outputs=out_decoder)

    encoder.summary()
    decoder.summary()
    input_auto = K.Input(shape=input_dims)
    encoderOut = encoder(input_auto)
    decoderOut = decoder(encoderOut)
    auto = K.models.Model(inputs=input_auto, outputs=decoderOut)
    auto.compile(optimizer='Adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
