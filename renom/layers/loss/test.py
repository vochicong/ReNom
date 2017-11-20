import renom as rm
import numpy as np
from renom.core import to_value, Variable
from renom.layers.loss.smoothed_l1 import smoothed_l1
from renom.cuda import set_cuda_active

set_cuda_active(True)
y = np.array([[-1, 1], [-1, 1]])
x = np.array([[1, -1], [1, -1]])
print(smoothed_l1(x, y))
