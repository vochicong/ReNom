#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
import numpy as np
import tempfile
import zlib
import struct
import threading
import queue


class DecompThread(threading.Thread):
    QUEUE = queue.Queue()

    def run(self):
        while True:
            q = self.QUEUE.get()
            q()

    @classmethod
    def submit(cls, rec):
        ev = threading.Event()

        def run():
            ev.ret = zlib.decompress(rec)
            ev.set()

        cls.QUEUE.put(run)
        return ev


class ReplayBuffer:
    THREAD_STARTED = False
    # todo: adjust buf size

    def __init__(self, action_shape, state_shape, buffer_size=1e5):
        self._f = tempfile.TemporaryFile()
        self._toc = {}
        self._action_size = np.prod(action_shape) if hasattr(
            action_shape, "__getitem__") else action_shape
        self._state_size = np.prod(state_shape) if hasattr(
            state_shape, "__getitem__") else state_shape

        self._action_shape = action_shape
        self._state_shape = state_shape

        self._size_prestate = self._state_size * np.float32().itemsize
        self._size_action = self._action_size * np.float32().itemsize
        self._size_reward = np.float32().itemsize
        self._size_state = self._state_size * np.float32().itemsize
        self._size_terminal = self._state_size * 1

        self._pos_action = 0 + self._size_prestate
        self._pos_reward = self._pos_action + self._size_action
        self._pos_state = self._pos_reward + self._size_reward
        self._pos_terminal = self._pos_state + self._size_state

        if not ReplayBuffer.THREAD_STARTED:
            # todo: adjust number of threads
            for _ in range(4):
                d = DecompThread()
                d.daemon = True
                d.start()
            ReplayBuffer.THREAD_STARTED = True

        self._buffer_size = int(buffer_size)
        self._count = 0
        self._full = False

    def store(self, prestate, action, reward, state, terminal):
        self.add(self._count, prestate, action, reward, state, terminal)
        self._count += 1
        if self._count >= self._buffer_size:
            self._count = 0
            self._full = True

    def add(self, index, prestate, action, reward, state, terminal):
        # todo: reuse buf when overwriting to the same index
        self._f.seek(0, 2)

        start = self._f.tell()
        c = zlib.compressobj(1)
        self._f.write(c.compress(prestate.astype('float32').tobytes()))
        self._f.write(c.compress(action.astype('float32').tobytes()))
        self._f.write(c.compress(struct.pack('f', reward)))
        self._f.write(c.compress(state.astype('float32').tobytes()))
        self._f.write(c.compress(terminal.tobytes()))
        self._f.write(c.flush())
        end = self._f.tell()
        self._toc[index] = (start, end)

    def _readrec(self, index):
        f, t = self._toc[index]
        self._f.seek(f, 0)
        rec = self._f.read(t - f)
        return rec

    def _unpack(self, buf):
        prestate = np.frombuffer(buf, np.float32, self._state_size, 0)
        action = np.frombuffer(buf, np.float32, self._action_size, self._pos_action)
        reward = struct.unpack('f', buf[self._pos_reward:self._pos_reward + self._size_reward])[0]
        state = np.frombuffer(buf, np.float32, self._state_size, self._pos_state)
        terminal = buf[self._pos_terminal]
        return prestate, action, reward, state, terminal

    def get(self, index):
        buf = self._readrec(index)
        buf = zlib.decompress(buf)
        return self._unpack(buf)

    def get_minibatch(self, batch_size=32, shuffle=True):
        perm = np.random.permutation(len(self))[:batch_size] if shuffle else np.arange(batch_size)
        n = len(perm)
        prestates = np.empty((n, self._state_size), dtype=np.float32)
        actions = np.empty((n, self._action_size), dtype=np.float32)
        rewards = np.empty(n, dtype=np.float32)
        states = np.empty((n, self._state_size), dtype=np.float32)
        terminals = np.empty(n, dtype=np.bool)

        events = []
        for index in perm:
            buf = self._readrec(index)
            events.append(DecompThread.submit(buf))

        for i, ev in enumerate(events):
            ev.wait()
            prestate, action, reward, state, terminal = self._unpack(ev.ret)
            prestates[i] = prestate
            actions[i] = action
            rewards[i] = reward
            states[i] = state
            terminals[i] = terminal
        shape = [n, ] + list(self._state_shape)
        prestates = prestates.reshape(shape)
        states = states.reshape(shape)
        return prestates, actions, rewards, states, terminals

    def __len__(self):
        return self._count if not self._full else self._buffer_size


memmap_path = ".replay_buf"
