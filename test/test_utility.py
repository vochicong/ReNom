#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import renom.cuda as cuda
from renom.utility.reinforcement.replaybuffer import ReplayBuffer

skipgpu = pytest.mark.skipif(not cuda.has_cuda(), reason="cuda is not installed")
skipmultigpu = pytest.mark.skipif(cuda.cuGetDeviceCount() < 2,
                                  reason="Number of gpu card is less than 2.")

np.random.seed(9)

eps = np.sqrt(np.finfo(np.float32).eps)


def auto_diff(function, node, *args):
    loss = function(*args)
    return loss.grad().get(node)


def numeric_diff(function, node, *args):
    shape = node.shape
    diff = np.zeros_like(node)

    if True:  # 5 point numerical diff
        coefficients1 = [1, -8, 8, -1]
        coefficients2 = [-2, -1, 1, 2]
        c = 12
    else:    # 3 point numerical diff
        coefficients1 = [1, -1]
        coefficients2 = [1, -1]
        c = 2

    node.to_cpu()
    node.setflags(write=True)
    for nindex in np.ndindex(shape):
        loss = 0
        for i in range(len(coefficients1)):
            dx = eps * node[nindex] if node[nindex] != 0 else eps
            node[nindex] += coefficients2[i] * dx
            node.to_cpu()
            ret_list = function(*args) * coefficients1[i]
            ret_list.to_cpu()
            node[nindex] -= coefficients2[i] * dx
            loss += ret_list
            loss.to_cpu()

        v = loss / (dx * c)
        v.to_cpu()
        diff[nindex] = v
    diff.to_cpu()
    return diff


@pytest.mark.parametrize("action_size, state_size, data_size", [
    [(2, ), (3, ), 4],
    [(2, ), (3, 2), 4],
])
def test_replaybuffer(action_size, state_size, data_size):
    buffer = ReplayBuffer(action_size, state_size, data_size)
    a, p, s, r, t = [[] for _ in range(5)]
    for i in range(data_size):
        a.append(np.random.rand(*action_size))
        p.append(np.random.rand(*state_size))
        s.append(np.random.rand(*state_size))
        r.append(np.random.rand(1))
        t.append(np.random.rand(1).astype(np.bool))
        buffer.store(p[-1], a[-1], r[-1], s[-1], t[-1])

    data = buffer.get_minibatch(data_size, False)
    for i in range(data_size):
        assert np.allclose(p[i], data[0][i])
        assert np.allclose(a[i], data[1][i])
        assert np.allclose(r[i], data[2][i])
        assert np.allclose(s[i], data[3][i])
        assert np.allclose(t[i], data[4][i])
