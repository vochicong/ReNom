import numpy as np
import renom as rm


class CrossValidator():

    def __init__(self):
        pass

    def validate(self, trainer, train_distributor, test_distributor, k=4):
        data_size = len(train_distributor)
        one_data_size = data_size // k
        results = []
        for i in range(k):
            train_dist, test_dist = train_dist[i * one_data_size:(i + 1) * one_data_size]
            trainer.train(train_dist, test_dist)
            results.append(trainer.test(test_dist))
        return np.mean(results)
