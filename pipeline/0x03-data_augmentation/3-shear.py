#!/usr/bin/env python3
""" Shear """
import tensorflow as tf


def shear_image(image, intensity):
    """ randomly shears an image:

        image is a 3D tf.Tensor containing the image to shear
        intensity is the intensity with which the image should be sheared

        Returns the sheared image
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared_img = tf.keras.preprocessing.image.random_shear(img, intensity)
    heared_img = tf.keras.preprocessing.image.array_to_img(sheared_img)
    return sheared_img
