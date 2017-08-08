import pytest


@pytest.fixture(params=[True, False])
def use_gpu(request):
    """
    Gpu switch for test.
    """
    yield request.param
