#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import renom as rm


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class VGG16(rm.Model):

    def __init__(self, classes=10):
        self._block1 = layer_factory(channel=64, conv_layer_num=2)
        self._block2 = layer_factory(channel=128, conv_layer_num=2)
        self._block3 = layer_factory(channel=256, conv_layer_num=3)
        self._block4 = layer_factory(channel=512, conv_layer_num=3)
        self._block5 = layer_factory(channel=512, conv_layer_num=3)
        self._dense1 = rm.Dense(4096)
        self._dense2 = rm.Dense(4096)
        self._dense3 = rm.Dense(classes)

    def forward(self, x):
        h = self._block1(x)
        h = self._block2(h)
        h = self._block3(h)
        h = self._block4(h)
        h = self._block5(h)
        h = rm.flatten(h)
        h = rm.relu(self._dense1(h))
        h = rm.relu(self._dense2(h))
        z = self._dense3(h)
        return z
