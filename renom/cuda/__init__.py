import traceback
import contextlib
import warnings
try:
    from renom.cuda.cuda_base import *
    from renom.cuda.cublas import *
    from renom.cuda.thrust import *
    _has_cuda = True
except ImportError as e:
    _has_cuda = False

_cuda_is_active = False
_cuda_is_disabled = False


def set_cuda_active(activate=True):
    '''If True is given, this method activate cuda.

    Args:
        activate (bool): Cuda activation flag.

    '''
    global _cuda_is_active
    if not has_cuda():
        warnings.warn("Couldn't find cuda modules.")
    _cuda_is_active = activate


def is_cuda_active():
    """Check the CUDA active.

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
