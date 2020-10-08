#!/usr/bin/env python3
""" Hue """
import tensorflow as tf


def change_hue(image, delta):
    """ changes the hue of an image:

        image is a 3D tf.Tensor containing the image to change
        delta is the amount the hue should change

        Returns the altered image
    """
    img = tf.image.adjust_hue(image, delta)
    return img
