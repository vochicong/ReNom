#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
                 optimizer=None, shuffle=True, events=None):

        self.model = model
        self.num_epoch = num_epoch
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.shuffle = shuffle

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

        while self.epoch < self.num_epoch:
            self.on_event('start_epoch')
            self.nth = 0

            for data, target in self.distributor.batch(self.batch_size, self.shuffle):
                self.data = data
                self.target = target

                self.on_event('forward')
                with self.model.train():
                    self.output = self.model(self.data)
                    self.loss = self.loss_func(self.output, self.target)

                self.on_event('backward')
                self.grads = self.loss.grad()
                self.grads.update(self.optimizer)

                self.on_event('updated')
                self.nth += 1

                # release objects
                self.data = self.target = None
                self.output = self.loss = self.grads = None

            self.on_event('end_epoch')
            self.epoch += 1

    def test(self, data):
        ret = self.model(data)
        return ret
