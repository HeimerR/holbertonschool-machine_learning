#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs a convolution on images with channels:

    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[0]
    conv_h = int(((h-kh)/sh) + 1)
    conv_w = int(((w-kw)/sw) + 1)
    conv = np.zeros((m, conv_h, conv_w, c))
    img = np.arange(0, m)
    for j in range(conv_h):
        for i in range(conv_w):
            if mode == 'max':
                conv[img, j, i] = (np.max(images[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
            if mode == 'avg':
                conv[img, j, i] = (np.mean(images[img,
                                   j*sh:(kh+(j*sh)),
                                   i*sw:(kw+(i*sw))], axis=(1, 2)))
    return conv
