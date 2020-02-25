#!/usr/bin/env python3
""" Train """
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """  builds, trains, and saves a neural network classifier """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    train_op_v = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(iterations):
        cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
        cost_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if i % 100 == 0:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(acc))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
        sess.run(train_op,  feed_dict={x: X_train, y: Y_train})
    saver.save(sess, save_path)
    return save_path
