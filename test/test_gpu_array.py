#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import pytest
from renom.cuda import set_cuda_active, use_cuda, disable_cuda, use_device
from renom.core import to_value, Variable, get_gpu
from renom.operation import dot, sum, sqrt
from renom.config import precision
import renom as rm
import test_utility
from renom.layers.function.batch_normalize import BATCH_NORMALIZE_FEATUREMAP


# if precision is not np.float32:
#    pytestmark = pytest.mark.skip()


def rand(shape):
    return np.array(np.random.rand(*shape), dtype=precision)


def randInt(shape):
    return np.array(np.random.randint(0, 2, shape), dtype=precision)


def arange(shape):
    return np.arange(np.prod(shape), dtype=precision).reshape(shape)


def close(a, b):
    assert np.allclose(to_value(a), to_value(b), atol=1e-4, rtol=1e-3)


@test_utility.skipgpu
def test_gpu_node_neg():
    set_cuda_active(True)
    a = np.array(np.random.rand(10, )).astype(precision)

    g1 = Variable(a)
    g2 = -g1
    g2.to_cpu()

    set_cuda_active(False)
    close(g2, -g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_add(a, b):
    with use_cuda():

        g1 = Variable(a)
        g2 = Variable(b)

        g3 = rm.sum(g1 + g2)
        g = g3.grad()

        g_g1 = g.get(g1)
        g_g2 = g.get(g2)
        g3.to_cpu()

    c3 = rm.sum(g1 + g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_mul(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 * g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 * g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_sub(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 - g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 - g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((2, 3))],
    [rand((2, 3)), rand((3,))],
    [rand((3,)), rand((2, 3))],
    [rand((2, 3, 3)), rand((3, ))],
    [rand((2, 3, 3)), rand((3, 3, ))],
])
def test_gpu_node_div(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 / g2)
    g = g3.grad()

    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 / g2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)

    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
    rand((2, 3)),
    rand((3,)),
    rand((2, 3, 3, 1)),
    rand((2, 3, 3, 1))
])
def test_gpu_node_abs(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(abs(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(abs(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), rand((3, 3))],
    [rand((2, 1)), rand((1, 3))],
    [rand((1, 1)), rand((1, 3))],
    [rand((1, 1)), rand((1, 1))],
])
def test_gpu_node_dot(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = dot(g1, g2)
    g4 = rm.sum(g3)
    g = g4.grad()
    g_g1 = g.get(g1)
    g_g2 = g.get(g2)
    g_g3 = g.get(g3)
    g3.to_cpu()
    g4.to_cpu()

    set_cuda_active(False)
    c3 = dot(g1, g2)
    c4 = rm.sum(c3)
    c = c4.grad()
    c_g1 = c.get(g1)
    c_g2 = c.get(g2)
    c_c3 = c.get(c3)

    close(g3, c3)
    close(g4, c4)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_c3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((2, 3)), rand((1, ))],
    [rand((3,)), rand((1, ))],
])
def test_gpu_node_pow(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sum(g1 ** g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1 ** g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)

# NEED to fix. Axis param


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((2, 3)),
    arange((2, 1, 3)),
])
def test_gpu_node_sum(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(g1)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(g1)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)

# NEED to fix. Axis param


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((2, 3)),
    rand((2, 1)),
    # rand((2, 1, 2, 2))  # error
])
def test_gpu_node_sum_axis(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(sum(g1, axis=0))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(sum(g1, axis=0))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((2, 3)),
    rand((2, 3, 3, 4)),
])
def test_gpu_node_square(a):
    set_cuda_active(True)

    g1 = Variable(a)
    g3 = sum(sqrt(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(sqrt(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
    [rand((1, 3, 3, 3)), randInt((1, 3, 3, 3))],
])
def test_gpu_node_softmax_cross_entropy(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.softmax_cross_entropy(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.softmax_cross_entropy(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
])
def test_gpu_node_sigmoid_cross_entropy(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.sigmoid_cross_entropy(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sigmoid_cross_entropy(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a, b", [
    [rand((1, 3)), randInt((1, 3))],
    [rand((2, 3)), randInt((2, 3))],
    [rand((2, 3, 3)), randInt((2, 3, 3))],
])
def test_gpu_node_mean_squared_error(a, b):
    set_cuda_active(True)

    g1 = Variable(a)
    g2 = Variable(b)

    g3 = rm.mean_squared_error(g1, g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.mean_squared_error(g1, g2)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_sigmoid(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.sigmoid(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.sigmoid(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_relu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.relu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.relu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_selu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.selu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.selu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_elu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.elu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.elu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_leaky_relu(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = sum(rm.leaky_relu(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = sum(rm.leaky_relu(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_tanh(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.tanh(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.tanh(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_softmax(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.softmax(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.softmax(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@pytest.mark.skip()
@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_dropout(a):
    set_cuda_active(True)

    g1 = Variable(a)

    layer = rm.Dropout()

    np.random.seed(1)
    g3 = rm.sum(layer(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    np.random.seed(1)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@pytest.mark.skip()
@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((4, 3, 3, 3)),
])
def test_gpu_node_spatial_dropout(a):
    with use_cuda():

        g1 = Variable(a)

        layer = rm.SpatialDropout()

        np.random.seed(1)
        g3 = rm.sum(layer(g1))
        g = g3.grad()
        g_g1 = g.get(g1)
        g3.to_cpu()

    np.random.seed(1)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 1)),
    rand((1, 2)),
    rand((3, 3)),
    rand((3, 1)),
    rand((10, 9))
])
def test_gpu_dense(a):
    layer = rm.Dense(output_size=2)

    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(layer(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(layer(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 2)),
    rand((2, 2)),
])
def test_gpu_lstm(a):
    layer = rm.Lstm(output_size=2)

    def func(x):
        loss = 0
        for _ in range(5):
            loss += sum(layer(x))
        layer.truncate()
        return loss

    set_cuda_active(True)

    g1 = Variable(a)

    g3 = func(g1)
    g3.to_cpu()

    g = g3.grad()
    g_g1 = g.get(g1)
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = func(g1)
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_convolution2d(a):
    with use_cuda():

        layer = rm.Conv2d(channel=32)
        layer.params["w"] = rm.Variable(np.random.rand(32, 3, 3, 3))
        layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_deconvolution2d(a):
    with use_cuda():

        layer = rm.Deconv2d(channel=32)
        layer.params["w"] = rm.Variable(np.random.rand(3, 32, 3, 3))
        layer.params["b"] = rm.Variable(np.random.rand(1, 32, 1, 1))

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g1 = g.get(layer.params["w"])
        g_g2 = g.get(layer.params["b"])
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(layer.params["w"])
    c_g2 = c.get(layer.params["b"])
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9))
])
def test_gpu_node_max_pooling(a):
    with use_cuda():

        layer = rm.MaxPool2d()

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c3.grad()
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((3, 3, 3, 3)),
    rand((1, 3, 9, 9)),
    rand((2, 3, 9, 9)),
    rand((2, 3, 12, 9)),
    rand((2, 3, 12, 5))
])
def test_gpu_node_average_pooling(a):
    with use_cuda():

        layer = rm.AveragePool2d()

        g1 = Variable(a)
        g2 = layer(g1)
        g3 = rm.sum(g2)
        g = g3.grad()
        g_g3 = g.get(g1)
        g2.to_cpu()
        g3.to_cpu()

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c3.grad()
    c_g3 = g.get(g1)

    close(g2, c2)
    close(g3, c3)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((4, 2)),
])
def test_batch_normalize(a):
    layer = rm.Sequential([rm.BatchNormalize(momentum=0.1)])

    set_cuda_active(True)

    g1 = Variable(a)
    g2 = layer(g1)
    g3 = rm.sum(g2)
    g = g3.grad()
    g_g1 = g.get(g1)
    g_g2 = g.get(layer.l0.params["w"])
    g_g3 = g.get(layer.l0.params["b"])

    layer.set_models(inference=True)
    g4 = layer(g1)
    layer.set_models(inference=False)

    g2.to_cpu()
    g3.to_cpu()
    g4.to_cpu()
    g_g1.to_cpu()
    g_g2.to_cpu()
    g_g3.to_cpu()

    set_cuda_active(False)
    layer.l0._mov_mean = 0
    layer.l0._mov_std = 0

    c2 = layer(g1)
    c3 = rm.sum(c2)
    c = c3.grad()
    c_g1 = c.get(g1)
    c_g2 = g.get(layer.l0.params["w"])
    c_g3 = g.get(layer.l0.params["b"])

    layer.set_models(inference=True)
    c4 = layer(g1)
    layer.set_models(inference=False)

    close(g2, c2)
    close(g3, c3)
    close(g4, c4)
    close(c_g1, g_g1)
    close(c_g2, g_g2)
    close(c_g3, g_g3)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    arange((2, 2, 2, 2)),
])
def test_batch_normalize_featuremap(a):
    layer = rm.BatchNormalize(mode=BATCH_NORMALIZE_FEATUREMAP, momentum=0.1)

    set_cuda_active(True)

    g1 = Variable(a)

    for _ in range(10):
        g3 = layer(g1)
    g3.to_cpu()

    layer.set_models(inference=True)
    g4 = layer(g1)
    layer.set_models(inference=False)

    set_cuda_active(False)
    layer._mov_mean = 0
    layer._mov_std = 0
    for _ in range(10):
        c3 = layer(g1)

    layer.set_models(inference=True)
    c4 = layer(g1)
    layer.set_models(inference=False)

    close(g3, c3)
    close(g4, c4)
    close(g3.attrs._m.new_array(), c3.attrs._m)
    close(g3.attrs._v.new_array(), c3.attrs._v)
    close(g3.attrs._mov_m.new_array(), c3.attrs._mov_m)
    close(g3.attrs._mov_v.new_array(), c3.attrs._mov_v)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_reshape(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.reshape(g1, shape=(-1, 1)))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.reshape(g1, shape=(-1, 1)))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("a", [
    rand((1, 3)),
    rand((1, 1)),
    rand((3, 1)),
    rand((1, 3, 3, 3)),
])
def test_gpu_node_flatten(a):
    set_cuda_active(True)

    g1 = Variable(a)

    g3 = rm.sum(rm.flatten(g1))
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(rm.flatten(g1))
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((2, 3, 3, 3)),
])
def test_lrn(node):
    layer = rm.Lrn()

    with use_cuda():
        g1 = Variable(node)

        g3 = rm.sum(layer(g1))
        g = g3.grad()
        g_g1 = g.get(g1)

        g3.to_cpu()
        g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(layer(g1))

    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((3, 2)),
    randInt((3, 2, 5, 1)),
])
def test_indexing(node):
    set_cuda_active(True)
    g1 = Variable(node)
    g3 = rm.sum(g1[1:2, -1])
    g = g3.grad()
    g_g1 = g.get(g1)
    g3.to_cpu()
    g_g1.to_cpu()

    set_cuda_active(False)
    c3 = rm.sum(g1[1:2, -1])
    c = c3.grad()
    c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
@pytest.mark.parametrize("node", [
    randInt((3, 2)),
    randInt((3, 2, 5, 1)),
])
def test_where(node):
    #    set_cuda_active(is_active)

    with use_cuda():
        g1 = Variable(node)
        g3 = rm.sum(rm.where(g1 > 0.5, g1, 1))
        g = g3.grad()
        g_g1 = g.get(g1)
        g3.to_cpu()
        g_g1.to_cpu()

    with use_cuda():
        c3 = rm.sum(rm.where(g1 > 0.5, g1, 1))
        c = c3.grad()
        c_g1 = c.get(g1)

    close(g3, c3)
    close(c_g1, g_g1)


@test_utility.skipgpu
def test_copy_from_cpu():
    src = Variable(rand((100,)))

    dest = Variable(rand((100,)))
    dest.copy_from(src)

    close(src, dest)


@test_utility.skipgpu
def test_copy_from_gpu():
    set_cuda_active(True)

    src = Variable(rand((100,)))
    src.to_gpu()

    dest = Variable(rand((100,)))
    dest.to_gpu()

    dest.copy_from(src)
    close(src, dest)

    close(src._gpu.new_array(), dest._gpu.new_array())


@test_utility.skipmultigpu
def test_copy_from_another_gpu():
    set_cuda_active(True)

    src = Variable(rand((100,)))
    src.to_gpu()

    with use_device(1):
        dest = Variable(rand((100,)))
        dest.to_gpu()

    dest.copy_from(src)
    close(src, dest)

    close(src._gpu.new_array(), dest._gpu.new_array())
