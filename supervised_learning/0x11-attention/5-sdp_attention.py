#!/usr/bin/env python3
""" Scaled Dot Product Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ calculates the scaled dot product attention:

        - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        - mask is a tensor that can be broadcast into
            (..., seq_len_q, seq_len_v) containing the optional mask,
            or defaulted to None

        if mask is not None, multiply -1e9 to the mask and add it to the
            scaled matrix multiplication
        The preceding dimensions of Q, K, and V are the same

        Returns: output, weights
        output a tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention
        weights a tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) containing the attention weights
    """
    # (..., seq_len_q, seq_len_k)
    print("*"*50)
    print("Q", Q)
    print("*"*50)
    print("K", K)
    print("*"*50)
    print("V", V)
    print("*"*50)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    print("matmul_qk", matmul_qk)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    print("scaled_attention_logits", scaled_attention_logits)

    print("*"*50)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    print("attention_weighs", attention_weights)
    print("*"*50)
    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)
    print("ouput", output)
    print("*"*50)
    return output, attention_weights
