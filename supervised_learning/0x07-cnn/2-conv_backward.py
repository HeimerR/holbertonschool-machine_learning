#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def convolve(images, kernel):
    """ convolve simple same """
    h = images.shape[0]
    w = images.shape[1]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = int((kh - 1)/2)
    pw = int((kw - 1)/2)
    if kh % 2 == 0:
        ph = int(kh/2)
    if kw % 2 == 0:
        pw = int(kw/2)
    new_images = np.pad(images, pad_width=((ph, ph), (pw, pw)),
                        mode='symmetric')
    conv = np.zeros((h, w))
    for j in range(h):
        for i in range(w):
            conv[j, i] = (np.sum(new_images[j:kh+j, i:kw+i] *
                          kernel))
    return conv * -1


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:

    @dZ numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
        partial derivatives with respect to the unactivated output
        of the convolutional layer
        @m is the number of examples
        @h_new is the height of the output
        @w_new is the width of the output
        @c_new is the number of channels in the output
    @A_prev numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        @h_prev is the height of the previous layer
        @w_prev is the width of the previous layer
        @c_prev is the number of channels in the previous layer
    @W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
        @kh is the filter height
        @kw is the filter width
    @b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
    @padding is a string that is either same or valid,
        indicating the type of padding used
    @stride is a tuple of (sh, sw) containing the strides for the convolution
        @sh is the stride for the height
        @sw is the stride for the width
    Returns: the partial derivatives with respect to the
        previous layer (dA_prev), the kernels (dW),
        and the biases (db), respectively
    """
    m = A_prev.shape[0]

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "same":
        pad = (W.shape[0]-1)//2
    else:
        pad = 0
    x_padded = np.pad(A_prev,
                      ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                      'constant', constant_values=0)

    x_padded_bcast = np.expand_dims(x_padded, axis=-1)
    dZ_bcast = np.expand_dims(dZ, axis=-2)

    dW = np.zeros_like(W)
    f = W.shape[0]
    w_x = x_padded.shape[1]
    for a in range(f):
        for b in range(f):
            dW[a, b, :, :] = (np.sum(dZ_bcast *
                              (x_padded_bcast[:, a:w_x-(f-1-a),
                               b:w_x-(f-1-b), :, :]),
                              axis=(0, 1, 2)))

    dx = np.zeros_like(x_padded, dtype=float)
    Z_pad = f-1
    dZ_padded = (np.pad(dZ, ((0, 0), (Z_pad, Z_pad), (Z_pad, Z_pad),
                 (0, 0)), 'constant', constant_values=0))

    for m_i in range(A_prev.shape[0]):
        for k in range(W.shape[3]):
            for d in range(A_prev.shape[3]):
                temp = (convolve(dZ_padded[m_i, :, :, k], W[:, :, d, k]))
                dx[m_i, :, :, d] += temp[f//2:-(f//2), f//2:-(f//2)]
    dx = dx[:, pad:dx.shape[1]-pad, pad:dx.shape[2]-pad, :]

    return dx, dW, db
