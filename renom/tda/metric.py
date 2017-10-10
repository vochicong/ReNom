# -*- coding: utf-8 -*.t-
import numpy as np

from scipy.spatial import distance


class Distance(object):
    """Class of distance matrix.

    Params:
        metric：metric of distance matrix.

         For example, ‘cosine’, ‘euclidean’, ‘hamming’, ‘minkowski'.

    Example:
        >>> from tda.metric import Distance
        >>> metric = Distance(metric="euclidean")
    """

    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit_transform(self, data):
        """Function of creating distance matrix. This function return distance matrix.

        Params:
            data：raw data.
        """
        metric_list = ["braycurtis",
                       "canberra",
                       "chebyshev",
                       "cityblock",
                       "correlation",
                       "cosine",
                       "dice",
                       "euclidean",
                       "hamming",
                       "jaccard",
                       "kulsinski",
                       "mahalanobis",
                       "matching",
                       "minkowski",
                       "rogerstanimoto",
                       "russellrao",
                       "seuclidean",
                       "sokalmichener",
                       "sokalsneath",
                       "sqeuclidean",
                       "yule"]

        if data is None:
            raise Exception("Data must not None.")

        if type(data) is not np.ndarray:
            data = np.array(data)

        if self.metric not in metric_list:
            raise Exception("This metric is not usable.")

        dist_vec = distance.pdist(data, metric=self.metric)
        dist_matrix = distance.squareform(dist_vec)
        return dist_matrix
