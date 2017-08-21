#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.cuda import use_device

class _EventHandlers(object):
    def __init__(self, events):
        super(_EventHandlers, self).__setattr__('_events', events)

    def __getattr__(self, name):
        def deco(f):
            self._events[name] = f
            return f

        return deco

    def __setattr__(self, name, f):
        self._events[name] = f

    def get_handlers(self):
        return self._evnets


class Trainer(object):
    def __init__(self, model, num_epoch, loss_func, batch_size,
                 optimizer=None, shuffle=True, events=None, num_gpu=1):

        self.model = model
        self.num_epoch = num_epoch
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.shuffle = shuffle
        self.num_gpu = num_gpu

        if events:
            self._events = events.copy()
        else:
            self._events = {}

        self.events = _EventHandlers(self._events)

    def on_event(self, event):
        handler = self._events.get(event)
        if handler:
            handler(self)

    def train(self, distributor):
        self.epoch = 0

        self.distributor = distributor
        self.on_event('start')

        models = [self.model]
        if self.num_gpu > 1:
            models.extend([self.model.__class__() for _ in range(self.num_gpu-1)])
            for n in range(self.num_gpu):
                models[n].set_gpu(n)

        while self.epoch < self.num_epoch:
            self.on_event('start_epoch')
            self.nth = 0

            for data, target in self.distributor.batch(self.batch_size, self.shuffle):

                datalen = len(data) // len(models)
                self.data = [data[i:i + datalen] for i in range(0, len(data), datalen)]

                targetlen = len(target) // len(models)
                self.targets = [target[i:i + targetlen] for i in range(0, len(target), targetlen)]

                self.on_event('forward')
                self.outputs = []

                for gpu in range(self.num_gpu):
                    model = models[gpu]
                    with model.train():
                        self.outputs.append(model(self.data[gpu]))

                self.on_event('loss')
                self.losses = []

                for gpu in range(self.num_gpu):
                    model = models[gpu]
                    with use_device(gpu):
                        self.losses.append(self.loss_func(self.outputs[gpu], self.targets[gpu]))
                self.on_event('backward')
                self.grads = []

                for gpu in range(self.num_gpu):
                    model = models[gpu]
                    with use_device(gpu):
                        self.grads.append(self.losses[gpu].grad())

                if self.num_gpu > 1:
                    models[0].join_grads(self.grads[0], zip(models[1:], self.grads[1:]))

                self.grads[0].update(self.optimizer)

                self.on_event('updated')
                self.nth += 1

                # release objects
                self.data = self.target = None
                self.outputs = self.losses = self.grads = None

            self.on_event('end_epoch')
            self.epoch += 1

    def test(self, data):
        ret = self.model(data)
        return ret
