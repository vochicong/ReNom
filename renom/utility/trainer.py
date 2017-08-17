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

# Default events


def default_event_start(trainer):
    pass


def default_event_start_epoch(trainer):
    try:
        from tqdm import tqdm
        iter_count = len(trainer.train_distributor) / trainer.batch_size
        bar = tqdm(total=iter_count)
    except:
        import warnings
        warnings.warn(
            "To display progress bar, you need to install 'tqdm' module.")
        bar = None
    setattr(trainer, "bar", bar)


def default_event_forward(trainer):
    pass


def default_event_backward(trainer):
    pass


def default_event_updated(trainer):
    bar = getattr(trainer, "bar", None)
    if bar is not None:
        epoch = trainer.epoch
        train_loss = trainer.loss
        msg = "epoch%3d: loss %6.4f" % (epoch, train_loss)
        bar.set_description(msg)
        bar.update(1)


def default_event_end_epoch(trainer):
    epoch = trainer.epoch
    bar = getattr(trainer, "bar", None)
    test_distributor = trainer.test_distributor
    avg_train_loss = trainer.avg_train_loss
    avg_test_loss = 0
    msg = "epoch%3d: avg loss %6.4f" % (epoch, avg_train_loss)

    if test_distributor:
        for i, (data, target) in enumerate(test_distributor.batch(trainer.batch_size, trainer.shuffle)):
            test_loss = trainer.loss_func(trainer.model(data), target)
            avg_test_loss += (test_loss - avg_test_loss) / (i + 1)
        msg = "epoch%3d: avg loss %6.4f avg test loss %6.4f" % \
            (epoch, avg_train_loss, avg_test_loss)

    if bar is not None:
        bar.set_description(msg)
        bar.update(0)
        bar.refresh()
        bar.close()
    else:
        print(msg)


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
            self._events = {
                "start": default_event_start,
                "start_epoch": default_event_start_epoch,
                "forward": default_event_forward,
                "backward": default_event_backward,
                "updated": default_event_updated,
                "end_epoch": default_event_end_epoch
            }

        self.events = _EventHandlers(self._events)

    def on_event(self, event):
        handler = self._events.get(event)
        if handler:
            handler(self)

    def train(self, train_distributor, test_distributor=None):
        self.epoch = 0
        self.train_distributor = train_distributor
        self.test_distributor = test_distributor
        self.on_event('start')

        while self.epoch < self.num_epoch:
            self.on_event('start_epoch')
            self.nth = 0
            self.avg_train_loss = 0
            self.avg_test_loss = 0

            for i, (data, target) in enumerate(self.train_distributor.batch(self.batch_size, self.shuffle)):
                self.data = data
                self.target = target

                self.on_event('forward')
                with self.model.train():
                    self.output = self.model(self.data)
                    self.loss = self.loss_func(self.output, self.target)
                    self.avg_train_loss += (self.loss -
                                            self.avg_train_loss) / (i + 1)

                self.on_event('backward')
                self.grads = self.loss.grad()
                self.grads.update(self.optimizer)

                self.on_event('updated')
                self.nth += 1

                # release objects
                self.data = self.target = None
                self.output = self.loss = self.grads = None

            if self.test_distributor:
                for i, (data, target) in enumerate(self.test_distributor.batch(self.batch_size, self.shuffle)):
                    test_loss = self.loss_func(self.model(data), target)
                    self.avg_test_loss += (test_loss -
                                           self.avg_test_loss) / (i + 1)

            self.on_event('end_epoch')
            self.epoch += 1

            # release objects
            self.avg_test_loss = self.avg_train_loss = None

    def test(self, data):
        ret = self.model(data)
        return ret
