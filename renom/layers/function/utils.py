# -*- coding: utf-8 -*-
import numpy as np
precision = np.float32


def out_size(size, k, s, p):
    return (np.array(size) + np.array(p) * 2 - np.array(k)) // np.array(s) + 1


def transpose_out_size(size, k, s, p):
    return np.array(s) * (np.array(size) - 1) + np.array(k) - 2 * np.array(p)


def im2col(img, size, kernel, stride, padding, padwith=0.):
    N, channel, in_h, in_w = img.shape
    out_h, out_w = size
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    img_n = np.pad(img, ((0, 0), (0, 0), (p_h, p_h + s_h - 1),
                         (p_w, p_w + s_w - 1)), mode="constant", constant_values=padwith)
    col = np.ndarray((N, channel, k_h, k_w, out_h, out_w), dtype=precision)
    for i in range(k_h):
        iu = i + s_h * out_h
        for j in range(k_w):
            ju = j + s_w * out_w
            col[:, :, k_h - 1 - i, k_w - 1 - j, :,
                :] = img_n[:, :, i:iu:s_h, j:ju:s_w]
    return col


def col2im(col, size, stride, padding):
    in_h, in_w = size
    s_h, s_w = stride
    p_h, p_w = padding
    N, channel, k_h, k_w, out_h, out_w = col.shape
    img = np.zeros((N, channel, in_h + 2 * p_h + s_h - 1,
                    in_w + 2 * p_w + s_w - 1), dtype=precision)
    for i in range(k_h):
        iu = i + s_h * out_h
        for j in range(k_w):
            ju = j + s_w * out_w
            img[:, :, i:iu:s_h, j:ju:s_w] += col[:, :, k_h - 1 - i, k_w - 1 - j, :, :]
    im_shape = img.shape
    return img[:, :, p_h:im_shape[2] - (p_h + s_h - 1),
               p_w:im_shape[3] - (p_w + s_w - 1)]


def tuplize(x):
    return x if isinstance(x, tuple) else (x, x)
