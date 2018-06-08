# -*- coding: utf-8 -*-
import numpy as np
from renom.core import precision, to_value


def out_size(size, k, s, p):
    return ((np.array(size) + np.array(p) * 2 - np.array(k)) // np.array(s) + 1).astype(np.int)


def transpose_out_size(size, k, s, p):
    return (np.array(s) * (np.array(size) - 1) + np.array(k) - 2 * np.array(p)).astype(np.int)


def im2col(img, size, kernel, stride, padding, padWith=0.):
    N, channel, in_h, in_w = img.shape
    out_h, out_w = size
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    img_n = np.pad(img, ((0, 0), (0, 0), (p_h, p_h + s_h - 1),
                         (p_w, p_w + s_w - 1)), mode="constant", constant_values=padWith)
    col = np.ndarray((N, channel, k_h, k_w, out_h, out_w), dtype=precision)
    for i in range(k_h):
        iu = i + s_h * out_h
        for j in range(k_w):
            ju = j + s_w * out_w
            col[:, :, k_h - 1 - i, k_w - 1 - j, :,
                :] = img_n[:, :, i:iu:s_h, j:ju:s_w]
    return col

def imncol(img, weight, bias, stride, padding, padWith = 0.):
    N, in_channels, in_dims = img.shape[0], img.shape[1], img.shape[2:]
    out_channels = weight.shape[0]
    assert in_channels is weight.shape[1]
    dimensionality = len(in_dims)


    # Padding asks for (before, after) for each dimension or it generalizes the padding
    pad_tuple = (padding, padding + stride - 1)
    padded_image = np.pad(img, ((0, 0), (0, 0), *[pad_tuple for _ in range(dimensionality)]),
                        mode="constant", constant_values=padWith)
    ret = []
    for batch in range(N):
        tmp = []
        for out_channel in range(out_channels):
            tmp2 = 0
            for in_channel in range(in_channels):
                tmp2 += recursive_multiplication(img[batch,in_channel],weight[out_channel,in_channel],stride=stride)
            tmp.append(tmp2)
        ret.append(tmp)
    ret = np.array(ret)
    return np.array(ret)

def colnim(img, weight, bias, stride):
    print(img.shape)
    print(weight.shape)
    img = img.flatten()[::-1].reshape(*img.shape)
    ret = []
    for batch in range(img.shape[0]):
        tmp2 = 0
        for out_channel in range(weight.shape[0]):
            tmp = []
            for in_channel in range(weight.shape[1]):
                tmp.append(recursive_multiplication(img[batch,out_channel],weight[out_channel,in_channel],stride=1,offset=stride))
            tmp2 += np.array(tmp)
        ret.append(tmp2)
    ret = np.array(ret)
    ret = ret.flatten()[::-1].reshape(*ret.shape)
    return ret

def recursive_multiplication(A, B, stride=1, offset=0, pre_offset = 0, depth = 0):
    if len(A.shape) is 1:
        offset += pre_offset
        steps = (A.shape[0]+offset-B.shape[0])//stride+1
        ret = []
        l = len(B)
        if offset >= l:
            offset = l-1
        for o in range(1,offset+1,stride):
            ret.append(np.sum(A[0:o]*B[l-o:l]))
        for i in range((stride - offset) % stride, len(A)-len(B)+1, stride):
            ret.append(np.sum(A[i:i+len(B)]*B))
        l = len(A)
        for o in range(1,offset+1,stride):
            ret.append(np.sum(A[l-1-offset+o:l]*B[0:o]))
        #ret.append(calc_edge_after(A,B,stride,offset))
        return np.array(ret)
    else:
        if len(A) is 1:
            return np.array([recursive_multiplication(A[0],B[0],stride,offset,pre_offset,depth+1)])
        ret = []
        steps = (A.shape[0]+offset*2-B.shape[0])//stride+1

        l = len(B)
        if offset > 0:
            ret.append(recursive_multiplication(A[0:offset],B[l-offset:l],stride,offset-1,pre_offset+1,depth+1))
            #ret = np.concatenate([ret,recursive_multiplication(A[0:offset],B[l-offset:l],stride,offset-1,pre_offset+1)])
        ret = np.array(ret)
        if pre_offset > 0:
            return ret[0]
        if len(ret) is 1 and len(ret.shape) > 2:
            ret = ret[0]
        for i in range((stride - offset) % stride, len(A)-len(B)+1, stride):
            tmp = 0
            for k in range(B.shape[0]):
                if i+k < len(A)-1:
                    tmp += np.array(recursive_multiplication(A[i+k,...], B[k,...], stride, offset,pre_offset,depth+1))
            tmp = tmp.reshape(1,*tmp.shape)
            if not len(ret) is 0:
                ret = np.concatenate([ret,tmp])
            else:
                ret = tmp
        l = len(A)
        tmp = []
        if offset > 0:
            tmp.append(recursive_multiplication(A[l-offset:l],B[0:offset],stride,offset-1,pre_offset+1,depth+1))
            tmp = np.array(tmp)
            if len(tmp) is 1 and len(tmp.shape) > 2:
                tmp = tmp[0]
            ret = np.concatenate([ret,tmp])

        return np.array(ret)



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


def roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(np.floor(size * stride))
    end = int(np.ceil((size + 1) * stride))

    start = min(max((start + roi_offset), 0), max_size)
    end = min(max((end + roi_offset), 0), max_size)

    return slice(start, end), end - start


def roi_pooling_slice_decode(size, stride, out_size, roi_offset):
    start = int(np.floor(float(size - roi_offset) / stride))
    end = int(np.ceil(float(size - roi_offset + 1) / stride))

    start = min(max(start, 0), out_size)
    end = min(max(end, 0), out_size)
    return start, end


def region_cordinates(roi, spatial_scale):
    idx, xmin, ymin, xmax, ymax = to_value(roi)
    idx = int(idx)
    xmin = int(round(xmin * spatial_scale))
    ymin = int(round(ymin * spatial_scale))
    xmax = int(round(xmax * spatial_scale))
    ymax = int(round(ymax * spatial_scale))
    return idx, xmin, ymin, xmax, ymax
