import numpy as np
from renom.cuda.thrust import cubinarize
from renom.config import precision
from libc.stdint cimport uintptr_t
from cuda_utils import _VoidPtr

cudand_generator = None

def curand_check(curandStatus_t status):
    if status==CURAND_STATUS_SUCCESS:
        return
    else:
        raise Exception("An error has occurred in curand. Error code {}".format(status))

cdef createCurandGenerator(rng_type):
    cdef curandGenerator_t gen
    curand_check(curandCreateGenerator(&gen, rng_type))
    return <uintptr_t>gen

cdef destroyCurandGenerator(generator):
    cdef curandGenerator_t gen = <curandGenerator_t><uintptr_t>generator
    curand_check(curandDestroyGenerator(gen))
    return

class CuRandGen(object):

    def __init__(self, seed=None, rng_type=CURAND_RNG_PSEUDO_DEFAULT):
        self.r_type = rng_type
        self.set_seed(seed)

    def rand_uniform(self, gpu_value):
        cdef curandGenerator_t gen = <curandGenerator_t><uintptr_t>self.gen
        cdef void* ptr = <void*><uintptr_t>gpu_value._ptr
        if precision == np.float32:
            curandGenerateUniform(gen, <float*>ptr, <size_t>gpu_value.size)
        elif precision == np.float64:
            curandGenerateUniformDouble(gen, <double*>ptr, <size_t>gpu_value.size)
        else:
            raise Exception("{} is not supported precision".format(precision))
        return

    def rand_normal(self, gpu_value, mean=0.0, std=1.0):
        cdef curandGenerator_t gen = <curandGenerator_t><uintptr_t>self.gen
        cdef void* ptr = <void*><uintptr_t>gpu_value
        if precision == np.float32:
            curandGenerateNormal(gen, <float*>ptr, <size_t>gpu_value.size, <float>mean, <float>std)
        elif precision == np.float64:
            curandGenerateNormalDouble(gen, <double*>ptr, <size_t>gpu_value.size, <double>mean, <double>std)
        else:
            raise Exception("{} is not supported precision".format(precision))
        return

    def rand_bernoulli(self, gpu_value, prob=0.5):
        self.rand_uniform(gpu_value)
        cubinarize(gpu_value, prob, gpu_value)
        return

    def set_seed(self, seed):
        gen = getattr(self, "gen", None) 
        if gen:
            destroyCurandGenerator(gen)
        gen = createCurandGenerator(self.r_type)
        cdef curandGenerator_t generator = <curandGenerator_t><uintptr_t>gen
        curand_check(curandSetPseudoRandomGeneratorSeed(generator, <unsigned long long>seed))
        self.gen = gen
        return

    def __del__(self):
        destroyCurandGenerator(self.gen)
        return
