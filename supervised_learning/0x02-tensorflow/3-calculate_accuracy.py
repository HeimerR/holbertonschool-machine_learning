#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    #prediction = tf.argmax(prob, 1)
    equality = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy
