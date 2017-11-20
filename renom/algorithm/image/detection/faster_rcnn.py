from renom.cuda import *
from renom.core import Node, get_gpu, GPUValue
import renom as rm
import numpy as np

class Anchor(Node):
    def __new__(cls, base_size, ratios, scales, feat_stride, height, width):
        value = cls.calc_value(base_size, ratios, scales, feat_stride, height, width)
        ret = super(Anchor, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, base_size, ratios, scales, feat_stride, height, width):
        shift_x = np.arange(0, width) * feat_stride
        shift_y = np.arange(0, height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.stack((
        shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)
        K = shifts.shape[0]
        A = len(ratios) * len(scales)
        shifts = get_gpu(shifts)
        anchors = GPUValue(shape=(K, A, 4))
        ratios = get_gpu(np.array(ratios))
        scales = get_gpu(np.array(scales))
        cu_generate_anchors(shifts, base_size, ratios, scales, feat_stride, anchors)
       # anchors = rm.reshape(anchors, (K*A, 4))
        return anchors

class ClipRoi(Node):
    def __new__(cls, roi, start, end, step, min_v, max_v):
        value = cls.calc_value(roi, start, end, step, min_v, max_v)
        ret = super(ClipRoi, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, roi, start, end, step, min_v, max_v):
        roi = get_gpu(roi)
        ary = roi.zeros_like_me()
        cu_clip_roi(roi, start, end, step, min_v, max_v, ary)
        return ary

def clip_roi(roi, start, end, step, min_v, max_v):
    return ClipRoi(roi, start, end, step, min_v, max_v)

