# -*- coding: utf - 8 -*-
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


def pad_image(img, padding, stride, padWith=0.):
    dims = img.shape[2:]
    dimensionality = len(dims)
    pad_tuple = (padding, padding + stride - 1)
    pad_list = [(0,0), (0,0)]
    pad_list.extend([pad_tuple for _ in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    return padded_image


class MaskUnit:

    def __init__(self):
        self.output_positions = []
        self.weight_positions = []

    def add_output_position(self, out_pos):
        self.output_positions.append(out_pos)

    def add_weight_position(self, weight_pos):
        self.weight_positions.append(weight_pos)

    def evaluate(self, img, kernel):
        ret = 0
        for out, weight in zip(self.output_positions, self.weight_positions):
            ret += img[out] * kernel[weight]
        return ret


def create_mask_array(img):
    if len(img.shape) > 1:
        return np.array([create_mask_array(img[0]) for p in range(len(img))])
    else:
        return np.array([MaskUnit() for _ in range(len(img))])


def create_backward_mask(img, kernel, stride, mask_array, padding=0):
    for p_img in generate_positions(img, stride, min_space=np.array(kernel.shape) - 1, offset=padding):
        p_img = np.array(p_img)

        for p_kern, v_kern in np.ndenumerate(kernel):
            skip = False
            for p in p_img + p_kern:
                if p < 0:
                    skip = True
            if skip:
                continue
            try:
                mask_array[tuple(p_img + p_kern)
                           ].add_output_position(tuple((p_img + padding) // stride))
                mask_array[tuple(p_img + p_kern)].add_weight_position(p_kern)
            except IndexError:
                pass


def imncol(img, weight, bias, stride, padding, padWith=0.):
    N, in_channels, in_dims = img.shape[0], img.shape[1], img.shape[2:]
    out_channels = weight.shape[0]
    assert in_channels is weight.shape[1], "Number of feature maps is not the same for input and output"
    assert stride > 0, "Kernel must move with positive increments"
    dimensionality = len(in_dims)

    # Padding asks for (before, after) for each dimension or it generalizes the padding
    pad_tuple = (padding, padding + stride - 1)
    pad_list = [(0,0), (0,0)]
    pad_list.extend([pad_tuple for _ in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    ret = []
    for batch in range(N):
        tmp = []
        for out_channel in range(out_channels):
            tmp2 = 0
            for in_channel in range(in_channels):
                tmp2 += place_kernels(padded_image[batch, in_channel],
                                      weight[out_channel, in_channel], stride=stride, offset=0)
            tmp.append(tmp2)
        ret.append(tmp)
    ret = np.array(ret)
    return np.array(ret)


def imnpool(img, kernel, stride, padding, padWith=0):
    N, in_channels, in_dims = img.shape[0], img.shape[1], img.shape[2:]
    dimensionality = len(in_dims)
    assert stride > 0

    pad_tuple = (padding, padding + stride - 1)
    pad_list = [(0,0), (0,0)]
    pad_list.extend([pad_tuple for _ in range(dimensionality)])
    padded_image = np.pad(img, tuple(pad_list),
                          mode="constant", constant_values=padWith)
    ret = []
    indx = []
    for batch in range(N):
        tmp = []
        tmp2 = []
        for in_channel in range(in_channels):
            ret2 = place_pools(padded_image[batch, in_channel], kernel, stride)
            tmp.append(ret2[0])
            tmp2.append(ret2[1])
        ret.append(tmp)
        indx.append(tmp2)
    ret = np.array(ret)
    indx = np.array(indx)
    return ret, indx


def poolnim(original, dy, indices):
    ret = np.zeros_like(original)
    for batch in range(original.shape[0]):
        for in_channel in range(original.shape[1]):
            tmp = ret[batch, in_channel]
            tmp.setflags(write=True)
            for i in range(len(indices[batch, in_channel])):
                tmp_list = [batch,in_channel]
                tmp_list.extend(indices[batch,in_channel, i])
                ret[tuple(tmp_list)
                    ] += dy[batch, in_channel].ravel()[i]
    return ret


def place_pools(img, kernel, stride, offset=0):
    kernels = []
    for i in range(len(img.shape)):
        kernels.append((img.shape[i] - kernel + offset * 2) // stride + 1)
    kernels = np.empty(tuple(kernels))
    indices = []

    for pos in generate_positions(img, stride, offset, min_space=np.array(kernel) - 1):
        kern = calculate_pool(img, kernel, pos)
        kernels[tuple(np.array(pos) // stride)] = kern[0]
        indices.append(kern[1])
    return kernels, indices


def calculate_pool(img, kernel, pos):
    if len(img.shape) > 1:
        tmp = []
        indx = []
        for i in range(0, kernel):
            if pos[0] + i < len(img) and pos[0] + i >= 0:
                ret = calculate_pool(img[pos[0] + i], kernel, pos[1:])
                tmp.append(ret[0])
                indx.append(ret[1])
        arg = np.argmax(tmp, axis=0)
        return np.array(tmp[arg[0]]), np.append(pos[0] + arg[0], indx[arg[0]])
    else:
        img = img[pos[0]:pos[0] + kernel]
        return np.amax(img, axis=0, keepdims=True), np.argmax([img]) + pos[0]


def pad_dx(dx, original):
    ret = np.zeros_like(original)
    for p, v in np.ndenumerate(dx):
        ret[p] = v
    return ret


def colnim(img, original, weight, bias, stride, backward_mask):
    ret = []
    for batch in range(img.shape[0]):
        tmp2 = 0
        for out_channel in range(weight.shape[0]):
            tmp = []
            for in_channel in range(weight.shape[1]):
                tmp.append(generate_backwards(img[batch, out_channel],
                                              weight[out_channel, in_channel], backward_mask))
            tmp2 += np.array(tmp)
        ret.append(tmp2)
    ret = np.array(ret)
    ret2 = []
    for out_channel in range(weight.shape[0]):
        tmp2 = 0
        for batch in range(img.shape[0]):
            tmp = []
            for in_channel in range(weight.shape[1]):
                tmp.append(generate_weights(
                    img[batch, out_channel], original[batch, in_channel], weight[out_channel,
                                                                                 in_channel], backward_mask))
            tmp2 += np.array(tmp)
        ret2.append(tmp2)
    ret2 = np.array(ret2)

    return ret, ret2


def generate_backwards(img, kernel, mask_array):
    ret = np.empty_like(mask_array)
    for p, v in np.ndenumerate(mask_array):
        ret[p] = v.evaluate(img, kernel)
    return ret


def generate_weights(img, original, kernel, mask_array):
    ret = np.zeros_like(kernel)
    for p, v in np.ndenumerate(mask_array):
        for w, y in zip(v.weight_positions, v.output_positions):
            ret[w] += img[y] * original[p]
    return ret


def generate_positions(img, stride=1, offset=0, min_space=0):
    pos = []
    if not isinstance(min_space, np.ndarray):
        min_space_list = []
        for _ in range(len(img.shape)):
            min_space_list.append(min_space)
    else:
        min_space_list = min_space
    for _ in range(len(img.shape)):
        pos.append(0)
    pos = np.array(pos, dtype=int)
    for p in enum_positions(pos, 0, len(pos), img.shape, stride, offset, min_space_list):
        yield tuple(p)


def enum_positions(pos_list, index, length, dist, stride, offset, min_space=0):
    pos_list[index] = -offset
    for x in range(-offset, dist[0] + offset - min_space[0], stride):
        if index < length - 1:
            for pos in enum_positions(pos_list, index + 1, length, dist[1:], stride, offset, min_space[1:]):
                yield pos
        else:
            yield pos_list
        pos_list[index] += stride


def place_kernels(img, kernel, stride, offset):
    kernels = []
    for i in range(len(img.shape)):
        kernels.append((img.shape[i] - kernel.shape[i] + offset * 2) // stride + 1)
    kernels = np.empty(tuple(kernels))
    assert len(kernel) > offset, "{}\{}".format(len(kernel), offset)
    for pos in generate_positions(img, stride, offset, min_space=np.array(kernel.shape) - 1):
        slices = [slice(pos[i], pos[i] + kernel.shape[i]) for i in range(len(img.shape))]
        kern = np.sum(img[slices] * kernel)
        kernels[tuple(np.array(pos) // stride)] = kern
    return kernels


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
