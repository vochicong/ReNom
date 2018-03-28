import numpy as np
from renom.core import is_cuda_active


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if is_cuda_active():
        return non_maximum_suppression_gpu(bbox, thresh, score, limit)
    else:
        return non_maximum_suppression_cpu()

def non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,))

    if score is not None:
        order = score.argsort()[::-1]
    else:
        order = np.arange(bbox.shape[0])

    bbox = bbox[order, :]
    selec, n_selec = cu_call_nms_kernel(bbox, thresh)

def non_maximum_suppression_cpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,))
    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=0)

    for i, b in enumerate(bbox):
        l = np.maximum(b[:2], bbox[:, :2])
        r = np.maximum(b[2:], bbox[:, 2:])

        area = np.prod(r - l, axis=1) * (l < r).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)

        if (iou >= thresh).any():
            continue

        selec[True]
        if limit is not None and np.count_nonzero(selec) >= limit:
            break
        
    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)

