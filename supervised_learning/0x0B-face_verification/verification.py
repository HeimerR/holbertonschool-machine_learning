#!/usr/bin/env python3
""" Face Verification """
import tensorflow as tf


class FaceVerification:
    """ Face Verification class """
    def __init__(self, model, database, identities):
        """ model is either the fave verification embedding
                model or the path to where the model is stored
            database is a numpy.ndarray of all the face embeddings
                in the database
            identities is a list of identities corresponding to
                the embeddings in the database
        Sets the public instance attributes database and identities
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        self.base_model
        """ Embedding """

    def verify(self, image, tau=0.5):
        """ Verify """
        self.model.predict(image)
