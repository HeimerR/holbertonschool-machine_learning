#!/usr/bin/env python3
""" NTS """
import numpy as np
import tensorflow as tf


class NST:
    """ class NTS """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        - style_image - the image used as a style reference,
            stored as a numpy.ndarray
        - content_image - the image used as a content reference,
            stored as a numpy.ndarray
        - alpha - the weight for content cost
        - beta - the weight for style cost

        if style_image is not a np.ndarray with the shape (h, w, 3),
            raise a TypeError with the message:
            style_image must be a numpy.ndarray with shape (h, w, 3)
        if content_image is not a np.ndarray with the shape (h, w, 3),
            raise a TypeError with the message:
            content_image must be a numpy.ndarray with shape (h, w, 3)
        if alpha is not a non-negative number,
            raise a TypeError with the message:
            alpha must be a non-negative number
        if beta is not a non-negative number,
            raise a TypeError with the message:
            beta must be a non-negative number
        Sets Tensorflow to execute eagerly
        Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
        """
        ls = len(style_image.shape)
        sch = style_image.shape[2]
        lc = len(content_image.shape)
        cch = content_image.shape[2]
        if not isinstance(style_image, np.ndarray) and ls != 3 and sch != 3:
            msg = "style_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if not isinstance(content_image, np.ndarray) and lc != 3 and cch != 3:
            msg = "content_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if beta < 0:
            raise TypeError("beta must be a non-negative number")
        tf.enable_eager_execution()
        self.content_image = self.scale_image(content_image)
        self.style_image = self.scale_image(style_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels """
        li = len(image.shape)
        ich = image.shape[2]
        if not isinstance(image, np.ndarray) and li != 3 and ich != 3:
            msg = "image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        new_h = 512
        new_w = 512
        if image.shape[0] > image.shape[1]:
            new_w = int(image.shape[1] * 512 / image.shape[0])

        elif image.shape[0] < image.shape[1]:
            new_h = int(image.shape[0] * 512 / image.shape[1])

        mth = tf.image.ResizeMethod.BICUBIC
        img = tf.image.resize_image_with_pad(image,
                                             new_h,
                                             new_w,
                                             method=mth)

        img = tf.saturate_cast(img, dtype=tf.uint8)
        img = img / 255
        new_image = tf.expand_dims(img, 0)

        return new_image