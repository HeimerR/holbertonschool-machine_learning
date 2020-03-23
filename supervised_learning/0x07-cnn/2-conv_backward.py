#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


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
    Returns: the partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m = dZ[0]
    h_new = dZ[1]
    w_new = dZ[2]
    c_new = dz[3]
    h_prev = A_prev[0]
    w_prev = A_prev[1]
    c_prev = A_prev[2]
    kh = W[0]
    kw = W[1]
    sh = stride[0]
    sw = stride[1]
    tmp_W = W.copy()
    m = Y.shape[1]
    for ly in reversed(range(self.__L)):
	if ly == self.__L - 1:
	    dz = self.__cache["A"+str(ly+1)] - Y
	    dw = np.matmul(self.__cache["A"+str(ly)], dz.T) / m
	else:
	    d1 = np.matmul(tmp_W["W"+str(ly+2)].T, dzp)
	    if self.__activation == 'sig':
		d2 = (self.__cache["A"+str(ly+1)] *
		      (1-self.__cache["A"+str(ly+1)]))
	    else:
		d2 = 1-self.__cache["A"+str(ly+1)]**2
	    dz = d1 * d2
	    dw = np.matmul(dz, self.__cache["A"+str(ly)].T) / m
	db = np.sum(dz, axis=1, keepdims=True) / m
	if ly == self.__L - 1:
	    self.__weights["W"+str(ly+1)] = (tmp_W["W"+str(ly+1)] -
					     (alpha * dw).T)
	else:
	    self.__weights["W"+str(ly+1)] = (tmp_W["W"+str(ly+1)] -
					     (alpha * dw))
	self.__weights["b"+str(ly+1)] = tmp_W["b"+str(ly+1)] - alpha * db
	dzp = dz
