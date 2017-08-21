#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import matplotlib.pyplot as plt
import copy
import time
import numpy as np
from multiprocessing import Pipe, Process

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

import renom as rm
from renom.cuda import cuda, use_device, cuGetDeviceCount
from renom.optimizer import Sgd, Adam
from renom.core import DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom.operation import sum


DEBUG_GRAPH_INIT(True)

np.random.seed(10)

cuda.set_cuda_active(True)

mnist = fetch_mldata('MNIST original', data_home="dataset")

X = mnist.data
y = mnist.target

X = X.astype(np.float32)
X /= X.max()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
labels_train = LabelBinarizer().fit_transform(y_train).astype(np.float32)
labels_test = LabelBinarizer().fit_transform(y_test).astype(np.float32)


class MNist(rm.Model):
    def __init__(self):
        super(MNist, self).__init__()
        self.layer1 = rm.Dense(output_size=2000)
        self.layer2 = rm.Dense(output_size=10)

    def forward(self, x):
        return self.layer2(rm.relu(self.layer1(x)))


epoch = 10
batch = 10000
count = 0

learning_curve = []
test_learning_curve = []

opt = Adam()
opt = Sgd()

N = len(X_train)

NUM_GPU = cuGetDeviceCount() or 1

models = [MNist() for i in range(NUM_GPU)]
for x in range(NUM_GPU):
    models[x].set_gpu(x)

for i in range(epoch):
    start_t = time.time()
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        for x in range(1, NUM_GPU):
            models[x].dup(models[0])

        train_batches = []
        responce_batches = []

        for gpu in range(NUM_GPU):
            with use_device(gpu):
                t = X_train[perm[j * batch // NUM_GPU:(j + 1) * batch // NUM_GPU]]
                train_batches.append(t)

                r = labels_train[perm[j * batch // NUM_GPU:(j + 1) * batch // NUM_GPU]]
                responce_batches.append(r)

        results = []
        for gpu in range(NUM_GPU):
            with models[gpu].train():
                result = models[gpu](train_batches[gpu])
                results.append(result)

        losses = []
        grads = []
        for gpu in range(NUM_GPU):
            with use_device(gpu):
                l = rm.softmax_cross_entropy(results[gpu], responce_batches[gpu])
                losses.append(l)

                grad = l.grad()
                grads.append(grad)

        models[0].join_grads(grads[0], zip(models[1:], grads[1:]))

        grads[0].update(opt)
        loss += losses[0]

    train_loss = loss / (N // batch)

    test_loss = rm.softmax_cross_entropy(models[0](X_test), labels_test)

    test_learning_curve.append(test_loss)
    learning_curve.append(train_loss)
    print("epoch %03d train_loss:%f test_loss:%f took time:%f" %
          (i, train_loss, test_loss, time.time() - start_t))

ret = models[0](X_test)
ret.to_cpu()
predictions = np.argmax(np.array(ret), axis=1)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

plt.hold(True)
plt.plot(learning_curve, linewidth=3, label="train")
plt.plot(test_learning_curve, linewidth=3, label="test_old")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend()
plt.show()

loss = None
nn = None
grad = None
test_learning_curve = learning_curve = None
DEBUG_NODE_STAT()
