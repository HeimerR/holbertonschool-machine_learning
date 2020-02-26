#!/usr/bin/env python3
""" Evaluate """
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ evaluates the output of a neural network """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        root = "/".join(save_path.split("/")[:-1]) + "/"
        saver.restore(sess, tf.train.latest_checkpoint(root))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        forward_prop = graph.get_tensor_by_name("layer_2/BiasAdd:0")
        loss = graph.get_tensor_by_name("softmax_cross_entropy_loss/value:0")
        accuracy = graph.get_tensor_by_name("Mean:0")
        feed_dict = {x: X, y: Y}
        fp = sess.run(forward_prop, feed_dict)
        ac = sess.run(accuracy, feed_dict)
        ls = sess.run(loss, feed_dict)
    return fp, ac, ls
