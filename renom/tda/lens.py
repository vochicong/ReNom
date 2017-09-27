# -*- coding: utf-8 -*.t-
import numpy as np

from sklearn import decomposition, manifold
from sklearn.model_selection import train_test_split

import renom as rm
from renom.optimizer import Sgd


class L1Centrality(object):
    """Class of L1 Centrality lens.
    """

    def __init__(self):
        pass

    def fit_transform(self, data):
        """Function of projection data to L1 centrality axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.sum(data, axis=1).reshape(data.shape[0], 1)


class LinfCentrality(object):
    """Class of projection data to L-infinity centrality axis.
    """

    def __init__(self):
        pass

    def fit_transform(self, data):
        """Function of projection data to L-inf Centrality axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.max(data, axis=1).reshape(data.shape[0], 1)


class GaussianDensity(object):
    """Class of projection data to gaussian density axis.

    Params:
        h: The width of kernel.
    """

    def __init__(self, h=0.3):
        if h == 0:
            raise Exception("Parameter h must not zero.")

        self.h = h

    def fit_transform(self, data):
        """Function of projection data to Gaussian Density axis.

        Params:
            data: distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        return np.sum(np.exp(-(data**2 / (2 * self.h))), axis=1).reshape(data.shape[0], 1)


class PCA(object):
    """Class of projection data to PCA components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.pca = decomposition.PCA(n_components=(max(self.components) + 1))

    def fit_transform(self, data):
        """Function of projection data to PCA axis.

        Params:
            data: raw data or distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        projected_data = self.pca.fit_transform(data)
        self.contribution_ratio = self.pca.explained_variance_ratio_[self.components]
        self.axis = self.pca.components_
        return projected_data[:, self.components]


class TSNE(object):
    """Class of projection data to TSNE components axis.

    Params:
        components: The axis of projection. If you use compoents 0 and 1, this is [0, 1].
    """

    def __init__(self, components=[0, 1]):
        if components is None:
            raise Exception("Component error.")

        self.components = components
        self.tsne = manifold.TSNE(n_components=max(components) + 1, init='pca')

    def fit_transform(self, data):
        """Function of projection data to TSNE axis.

        Params:
            data: raw data or distance matrix.
        """
        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        projected_data = self.tsne.fit_transform(data)
        return projected_data[:, self.components]


class AutoEncoderNetwork(rm.Model):
    """Class of Auto Encoder Network.

    Params:
        unit_size: unit_size is the size of input data.
    """

    def __init__(self, unit_size):
        self._layer1 = rm.Dense(10)
        self._encodedlayer = rm.Dense(2)
        self._layer2 = rm.Dense(10)
        self._outlayer = rm.Dense(unit_size)

    def forward(self, x):
        """Forwarding.

        Params:
            x: training data.
        """
        l1_out = rm.sigmoid(self._layer1(x))
        l = rm.sigmoid(self._encodedlayer(l1_out))
        l2_out = rm.sigmoid(self._layer2(l))
        g = self._outlayer(l2_out)
        loss = rm.mse(g, x)
        return loss


class AutoEncoder(object):
    """Class of Auto Encoder dimention reduction lens.

    Params:
        epoch: training epoch.

        batch_size: batch size.

        opt: training optimizer.

        verbose: print message or not.
    """

    def __init__(self, epoch, batch_size, opt=Sgd(), verbose=0):
        self.epoch = epoch
        self.batch_size = batch_size
        self.opt = opt
        self.verbose = verbose

    def fit_transform(self, data):
        """dimention reduction function.

        Params:
            data: raw data or distance matrix.
        """
        n = data.shape[0]

        train_data, test_data = train_test_split(data, test_size=0.1, random_state=10)

        # TODO
        # ネットワークを引数で入れられるように変更する。
        self.network = AutoEncoderNetwork(data.shape[1])

        for i in range(self.epoch):
            total_loss = 0

            for ii in range(int(n / self.batch_size)):
                batch = data[ii * self.batch_size: (ii + 1) * self.batch_size]

                with self.network.train():
                    loss = self.network(batch)
                    loss.grad().update(self.opt)

                total_loss += loss
            if self.verbose == 1:
                print("epoch:{} loss:{}".format(i, total_loss / (n / self.batch_size)))

        projected_data = self.network._encodedlayer(rm.sigmoid(self.network._layer1(data)))
        return projected_data
