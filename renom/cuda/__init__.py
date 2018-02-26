import numpy as np
import traceback
import contextlib
import warnings
try:
    from renom.cuda.cuda_base import *
    from renom.cuda.cublas import *
    from renom.cuda.thrust import *
    from renom.cuda.curand import *
    _has_cuda = True
except ImportError as e:
    curand_generator = None
    _has_cuda = False

_cuda_is_active = False
_cuda_is_disabled = False


def set_cuda_active(activate=True):
    '''If True is given, cuda will be activated.

    Args:
        activate (bool): Activation flag.
    '''
    global _cuda_is_active
    if not has_cuda() and activate:
        warnings.warn("Couldn't find cuda modules.")
    _cuda_is_active = activate


def is_cuda_active():
    """Checks whether CUDA is activated.

    Returns:
        True if cuda is active.
    """
    return _cuda_is_active and has_cuda() and not _cuda_is_disabled


def has_cuda():
    """This method checks cuda libraries are available.

    Returns:
        True if cuda is correctly set.
    """
    return _has_cuda


@contextlib.contextmanager
def use_cuda(is_active=True):
    # save cuda state
    cur = _cuda_is_active
    set_cuda_active(is_active)
    try:
        yield None
    finally:
        # restore cuda state
        set_cuda_active(cur)


@contextlib.contextmanager
def disable_cuda(is_disabled=True):
    global _cuda_is_disabled
    # save cuda state
    cur = _cuda_is_disabled
    _cuda_is_disabled = is_disabled
    try:
        yield None
    finally:
        # restore cuda state
        _cuda_is_disabled = cur


@contextlib.contextmanager
def use_device(device_id):
    active = is_cuda_active()

    if active:
        cur = cuGetDevice()
        cuSetDevice(device_id)  # switch dedice

    try:
        yield
    finally:
        if active:
            cuSetDevice(cur)   # restore device


_CuRandGens = {}


def curand_generator(seed=None):
    deviceid = cuGetDevice()
    if seed is None:
        seed = seed if seed else np.random.randint(4294967295, size=1)

    if deviceid in _CuRandGens:
        gen = _CuRandGens[deviceid]
        gen.set_seed(seed)
        return gen

    ret = CuRandGen(seed)
    _CuRandGens[deviceid] = ret
    return ret


def release_mem_pool():
    """This function releases GPU memory pool.
    """
    if gpu_allocator:
        gpu_allocator.release_pool
