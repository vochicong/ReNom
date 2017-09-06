from __future__ import division, print_function
import os
import sys
import cPickle
import time


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report


import renom as rm
from renom.optimizer import Sgd, Adam
from renom.cuda.cuda import set_cuda_active, cuGetDeviceCount, cuDeviceSynchronize
from renom import cuda


from renom.utility.trainer import Trainer
from renom.utility.distributor import NdarrayDistributor


set_cuda_active(True)


dir = "cifar-10-batches-py/"
paths = ["data_batch_1", "data_batch_2", "data_batch_3",

         "data_batch_4", "data_batch_5"]


def unpickle(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d


# Load train data.
data = map(unpickle, [os.path.join(dir, p) for p in paths])
train_x = np.vstack([d["data"] for d in data])
train_y = np.vstack([d["labels"] for d in data])

# Load test data.
data = unpickle(os.path.join(dir, "test_batch"))
test_x = np.array(data["data"])
test_y = np.array(data["labels"])

# Rehsape and rescale image.
train_x = train_x.reshape(-1, 3, 32, 32)
train_y = train_y.reshape(-1, 1)
test_x = test_x.reshape(-1, 3, 32, 32)
test_y = test_y.reshape(-1, 1)

train_x = train_x / 255.
test_x = test_x / 255.

# Binalize
labels_train = LabelBinarizer().fit_transform(train_y)
labels_test = LabelBinarizer().fit_transform(test_y)

# Change types.
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)
labels_train = labels_train.astype(np.float32)
labels_test = labels_test.astype(np.float32)

N = len(train_x)


class Cifar10(rm.Model):

    def __init__(self):
        super(Cifar10, self).__init__()
        c = 64
        self._l1 = rm.Conv2d(channel=c)
        self._l2 = rm.Conv2d(channel=c)
        self._l3 = rm.Conv2d(channel=64)
        self._l4 = rm.Conv2d(channel=64)
        self._l5 = rm.Dense(512)
        self._l6 = rm.Dense(10)
        self._sd = rm.SpatialDropout(dropout_ratio=0.25)
        self._pool = rm.MaxPool2d(filter=2, stride=2)

    def forward(self, x):
        t1 = rm.relu(self._l1(x))
        b = self._l2(t1)
        a = rm.relu(b)
        b = self._pool(a)
        t2 = self._sd(b)
        c = self._l3(t2)
        t3 = rm.relu(c)
        t4 = self._sd(self._pool(rm.relu(self._l4(t3))))
        t5 = rm.flatten(t4)
        t6 = rm.dropout(rm.relu(self._l5(t5)))
        t7 = self._l6(t6)
        return t7


# Number of GPUs.
num_gpu = 4  # cuGetDeviceCount() or 1

trainer = Trainer(Cifar10(), num_epoch=10, loss_func=rm.softmax_cross_entropy,
                  batch_size=2000, optimizer=Adam(), num_gpu=num_gpu)

N = len(train_x)
trainer.train(NdarrayDistributor(train_x, labels_train),
              NdarrayDistributor(test_x, labels_test))

predicted = np.argmax(trainer.test(test_x), axis=1)

print(confusion_matrix(predicted, test_y))
print(classification_report(predicted, test_y))
