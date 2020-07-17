#!/usr/bin/env python3
""" Extract Word2Vec """
import tensorflow.keras as K
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """  gets the converts the gensim word2vec model to a keras layer:

        model is a trained gensim word2vec models
        Returns: the trainable keras Embedding
    """
    e = K.layers.Embedding(
    			   input_dim,
    			   utput_dim,
			   embeddings_initializer="uniform",
			   embeddings_regularizer=None,
			   activity_regularizer=None,
			   embeddings_constraint=None,
			   mask_zero=False,
			   input_length=None,
		    	   )


    return model.wv.get_keras_embedding(train_embeddings=True)
