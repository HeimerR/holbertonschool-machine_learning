#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    ac, _ = tf.metrics.accuracy(predictions=y_pred, labels=y)
    return tf.math.reduce_mean(_)
