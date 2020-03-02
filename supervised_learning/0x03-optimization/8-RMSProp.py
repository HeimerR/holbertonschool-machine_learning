#!/usr/bin/env python3
""" RMSProp Upgraded """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a NN
    in tensorflow using the RMSProp optimization algorithm
    """
    c = (tf.train.RMSPropOptimizer(alpha, momentum=beta2, epsilon=epsilon)
         .minimize(loss))
    return c
