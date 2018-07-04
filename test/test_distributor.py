import renom as rm
import numpy as np
import pytest
from renom.utility.distributor.distributor import *
from renom.cuda import *
import test_utility
from renom.config import precision


@test_utility.skipgpu
@pytest.mark.parametrize("data_shape", [
    (int(1e2), int(1e2)),  # Small Data Test
    (int(1e5), int(1e1)),  # Medium Data Test
    (int(1e8), int(1e0)),  # Large Data Test
])
def test_gpu_distributor(data_shape):
    # Construct the data from numpy
    X = np.full(data_shape, 1)
    X = np.arange(np.prod(data_shape)).reshape(*data_shape)
    X = X.astype(precision)
    #Y = (X*2).reshape(*data_shape)
    Y = np.full(data_shape, 0)
    batch_size = (len(X) // 100)
    #batch_size = len(X)
    # Set up the distributor and loop parameters
    set_cuda_active(True)
    print(is_cuda_active())
    data_distributor = GPUDistributor(x=X, y=Y)
    epochs = 3

    # Construct a data to test against
    # If the resulting data is different from this, an error occured somewhere
    #test_result = np.full(data_shape, 3).astype(precision)
    #test_result = X+Y

    # Create a big pool of data for renom to re-use later, removing
    # potential delays from creating extra memory
    #pre_data = np.full(data_shape, 0)
    # get_gpu(pre_data)

    # Test the distributor over a normal loop to see if it always produces correct data.
    for e in range(epochs):
        i = 0
        for batch_x, batch_y in data_distributor.batch(batch_size, shuffle=False):
            test_result = X[i * batch_size:(i + 1) * batch_size] + \
                Y[i * batch_size:(i + 1) * batch_size]
            result = Node(batch_x + batch_y)
            result.to_cpu()
            result = result.as_ndarray()
            assert np.allclose(result, test_result), "\n{}".format(np.isclose(result, test_result))
            i += 1
