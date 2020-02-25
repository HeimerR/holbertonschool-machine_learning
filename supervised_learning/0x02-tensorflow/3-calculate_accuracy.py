#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    correct_prediction = tf.equal(tf.round(y_pred), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
