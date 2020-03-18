#!/usr/bin/env python3
""" Convolution with Channels """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ performs a convolution on images with channels:

        @images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            @m is the number of images
            @h is the height in pixels of the images
            @w is the width in pixels of the images
            @c is the number of channels in the image
        @kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
        @kh is the height of the kernel
        @kw is the width of the kernel
        @padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                @ph is the padding for the height of the image
                @pw is the padding for the width of the image
        @stride is a tuple of (sh, sw)
            @sh is the stride for the height of the image
            @sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[0]
    if padding == 'valid':
        new_h = h
        new_w = w
        new_images = images
    if padding == 'same':
        ph = int((kh - 1)/2)
        pw = int((kw - 1)/2)
    if type(padding) == 'tuple':
        ph = padding[0]
        pw = padding[1]
    if padding == 'same' or type(padding) == 'tuple':
        new_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                            mode='constant', constant_values=0)
        new_h = new_images.shape[1]
        new_w = new_images.shape[2]
    out_h = int(((new_h-kh)/sh) + 1)
    out_w = int(((new_w-kw)/sw) + 1)
    conv = np.zeros((m, out_h, out_w))
    #img = np.arange((m, new_h, new_w, c))
    channel = np.arange(c)
    img, _1, _2, ch = np.indices(new_images.shape)
    print(img.shape)
    #print(new_images[img, 0:32, 0:32, img])
    for j in range(out_h):
        for i in range(out_w):
            conv[img, j, i] = (np.sum(new_images[img,
                               j*sh:(kh+(j*sh)),
                               i*sw:(kw+(i*sw)), ch] *
                               kernel, axis=(1, 2, 3)))

    return conv
