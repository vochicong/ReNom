import numpy as np
from renom.core import precision

# TODO: Make it changable.
try:
    if precision is np.float32:
        from renom.cuda.thrust_float import *
    else:
        from renom.cuda.thrust_double import *
except ImportError:
    raise
