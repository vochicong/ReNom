import numpy as np
import renom as rm

class CrossValidator():

    def __init__(self):
        pass

    def validate(self, trainer, train_distributor, test_distributor, k=4):
        data_size = len(train_distributor)
        one_data_size = data_size//k
        for i in range(k):
            train_dist, test_dist = None, None
            trainer.train(train_distributor, test_distributor)
