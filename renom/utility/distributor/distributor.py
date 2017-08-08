#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import warnings
import numpy as np


class Distributor(object):
    '''Distributor class
    This is the base class of a data distributor.

    Args:
        x (ndarray): Input data.
        y (ndarray): Target data.
        path (string): Path to data.

    >>> import numpy as np
    >>> from renom.utility.distributor.distributor import NdarrayDistributor
    >>> x = np.random.randn(100, 100)
    >>> y = np.random.randn(100, 1)
    >>> distributor = NdarrayDistributor(x, y)
    >>> batch_x, batch_y = distributor.batch(10).next()
    >>> batch_x.shape
    (10, 100)
    >>> batch_y.shape
    (10, 1)

    '''

    def __init__(self, x=None, y=None, path=None, data_table=None):
        self._folder_path = path
        self._data_x = x
        self._data_y = y
        self._data_table = data_table
        self._data_size = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            return self.__class__(x=self._data_x[start:stop:step],
                                  y=self._data_y[start:stop:step],
                                  data_table=self._data_table)
        else:
            return self._data_x[index], self._data_y[index]

    def __len__(self):
        return self._data_size

    def batch(self, batch_size, shuffle=True):
        '''
        This function returns `minibatch`.

        Args:
            batch_size (int): Size of batch.
            shuffle (bool): If True is passed, data will be selected randomly.
        '''
        if shuffle:
            perm = np.random.permutation(self._data_size)
        else:
            perm = np.arange(self._data_size)
        for i in range(int(np.ceil(self._data_size / batch_size))):
            p = perm[i * batch_size:(i + 1) * batch_size]
            yield self._data_x[p], self._data_y[p]

    def kfold(self, num, overlap=False, shuffle=False):
        for i in range(num):
            yield self._data_x

    def split(self, ratio=0.8, shuffle=True):
        '''
        データを分割し、新たなDistributorインスタンスを作成する。

        Args:
            ratio (float): 分割比
            shuffle (boolean): 真のとき、分割時に並び順をシャッフルする。
        '''
        div = int(self._data_size * ratio)
        if shuffle:
            perm = np.random.permutation(self._data_size)
        else:
            perm = np.arange(self._data_size)

        for i in range(2):
            p = perm[div * i:div * (i + 1)]
            yield self.__class__(x=self._data_x[p],
                                 y=self._data_y[p],
                                 data_table=self._data_table)

    def data(self):
        return self._data_x, self._data_y

    @property
    def y(self):
        return self._data_y

    @property
    def x(self):
        return self._data_x


class NdarrayDistributor(Distributor):

    '''
    NumpyArrayを扱うDistributorクラス

    :param ndarray x: 入力データ
    :param ndarray y: 教師データ
    '''

    def __init__(self, x, y, **kwargs):
        super(NdarrayDistributor, self).__init__(x=x, y=y,
                                                 data_table=kwargs.get("data_table"))
        assert len(x) == len(y), "{} {}".format(len(x), len(y))
        self._data_size = len(x)

    def kfold(self, num=4, overlap=False, shuffle=True):
        if num < 2:
            warnings.warn(
                "If the argument 'num' is less than 2, it returns a pair of 'self' and 'None'.")
            yield self, None
            return
        div = int(np.ceil(self._data_size / num))
        flag = np.zeros((self._data_size))
        for i in range(num):
            flag[i * div:].fill(i)

        if shuffle:
            perm = np.random.permutation(self._data_size)
        else:
            perm = np.arange(self._data_size)

        for i in range(num):
            ty_flag = np.where(flag == i, False, True)
            ts_flag = np.where(flag == i, True, False)
            yield self.__class__(x=self._data_x[perm[ty_flag]],
                                 y=self._data_y[perm[ty_flag]],
                                 data_table=self._data_table),\
                self.__class__(x=self._data_x[perm[ts_flag]],
                               y=self._data_y[perm[ts_flag]],
                               data_table=self._data_table)


class TimeSeriesDistributor(NdarrayDistributor):

    '''
    時系列データを扱うDistributorクラス。
    時系列データは以下のフォーマットで与えられる必要がある。

    N: データ数
    T: 時系列長
    D: データ次元
    (N, T, D)

    :param ndarray x: 入力データ
    :param ndarray y: 教師データ
    '''

    def __init__(self, x, y, **kwargs):
        super(TimeSeriesDistributor, self).__init__(x=x, y=y,
                                                    data_table=kwargs.get("data_table"))
        assert x.ndim == 3
        assert len(x) == len(y)
        self._data_size = len(x)
