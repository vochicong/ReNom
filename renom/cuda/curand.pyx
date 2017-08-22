from curand import * 
import nunpy as np
from renom.config import precision
from libc.stdint cimport uintptr_t
from cuda_utils import _VoidPtr

def curand_check(curandStatus_t status):
    if status==CURAND_STATUS_SUCCESS:
        return
    else:
        raise Exception("An error occurred in curand.")

cdef createCurandGeneratorHost(rng_type):
    cdef curandGenerator_t *gen
    curand_check(curandCreateGeneratorHost(gen, rng_type))
    return <uintptr_t>gen

"""
class CuRandGen(object):
    def __init__(self, device_id, seed=None, rng_type=CURAND_RNG_PSEUDO_DEFAULT):
        self.seed = seed
        self.device_id = device_id
        self.gen = createCurandGeneratorHost(rng_type)
        if seed is not None:
            curand_check(curandSetPseudoRandomGeneratorSeed(self.gen, seed))

    def rand_uniform(self, size):
        curandGenerator_t gen = self.gen
        if precision == np.float32:
            curandGenerateUniform(gen)
        elif precision == np.float64:
            curandGenerateUniformDouble(gen)
        else:
            raise Exception("{} is not supported precision".format(precision))

    def rand_normal(self, size):
        gen = self.gen
        if precision == np.float32:
            curandGenerateNormal(gen)
        elif precision == np.float64:
            curandGenerateNormalDouble(gen)
        else:
            raise Exception("{} is not supported precision".format(precision))

    def rand_bernoulli(self, size, prob=0.5):
        pass

    def set_seed(self, seed):
        pass
"""

    
