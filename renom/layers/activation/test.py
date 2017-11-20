import renom as rm
from renom.cuda import set_cuda_active
import numpy as np
set_cuda_active(True)
x = np.random.rand(1, 3)
z = rm.softmax(x)
print(z)
