"""
ReNom
"""
from __future__ import absolute_import
from renom.layers.activation import *
from renom.layers.function import *
from renom.layers.loss import *
from renom.core import Variable
from renom.operation import *
from renom.optimizer import *
import numpy as np


def set_renom_seed(seed=30):
    if is_cuda_active():
        curand_set_seed(seed)
    np.random.seed(seed)


__version__ = "2.5.1"
