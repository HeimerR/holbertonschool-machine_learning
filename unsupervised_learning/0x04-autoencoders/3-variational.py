#!/usr/bin/env python3
""" Variational Autoencoder """
import tensorflow.keras as K


def sampling(args):
    """ sampling trick """

    z_mean, z_var = args
    m = K.backend.shape(z_mean)[0]
    dims = K.backend.int_shape(z_mean)[1]
    epsilon = K.backend.random_normal(shape=(m, dims))
    return z_mean + K.backend.exp(0.5 * z_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Creates a variational autoencoder:

        - input_dims is an integer containing the dimensions
            of the model input
        - hidden_layers is a list containing the number of
            nodes for each hidden layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of the
            latent space representation

        Returns: encoder, decoder, auto
            - encoder is the encoder model
            - decoder is the decoder model
            - auto is the full autoencoder model
        The autoencoder model should be compiled using adam optimization
        and binary cross-entropy loss
   	All layers should use a relu activation except for the mean and
	log variance layers in the encoder, which should use None, and the
	last layer in the decoder, which should use sigmoid
    """
    input_encoder = K.Input(shape=(input_dims, ))

    output = K.layers.Dense(hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        output = K.layers.Dense(hidden_layers[i], activation='relu')(output)

    z_mean = K.layers.Dense(latent_dims)(output)
    z_var = K.layers.Dense(latent_dims)(output)
    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_var])

    input_decoder = K.Input(shape=(latent_dims, ))
    output2 = K.layers.Dense(hidden_layers[-1],
                             activation='relu')(input_decoder)
    for i in range(len(hidden_layers)-2, -1, -1):
        output2 = K.layers.Dense(hidden_layers[i], activation='relu')(output2)
    out_decoder = K.layers.Dense(input_dims, activation='sigmoid')(output2)

    encoder = K.models.Model(inputs=input_encoder, outputs=[z, z_mean, z_var])
    decoder = K.models.Model(inputs=input_decoder, outputs=out_decoder)

    input_auto = K.Input(shape=(input_dims, ))
    encoderOut = encoder(input_auto)[0]
    decoderOut = decoder(encoderOut)
    auto = K.models.Model(inputs=input_auto, outputs=decoderOut)

    encoder.summary()
    decoder.summary()
    auto.summary()
    def loss(input_auto, decoderOut):
        """ custom loss function """
        reconstruction_loss = K.backend.binary_crossentropy(input_auto, decoderOut)
        reconstruction_loss = K.backend.sum(reconstruction_loss, axis=-1)
        print(reconstruction_loss.shape)
        kl_loss = - 0.5 * K.backend.sum(1 + z_var - K.backend.square(z_mean) - K.backend.exp(z_var), axis=-1)
        print(kl_loss.shape)
        return kl_loss + reconstruction_loss

    auto.compile(optimizer='Adam', loss=loss)

    return encoder, decoder, auto
