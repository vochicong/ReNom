import numpy as np
from renom.core import Variable, GPUValue, Node, to_value
from renom.config import precision
from renom.optimizer import Sgd
from renom.cuda import set_cuda_active

def close(a, b):
    print ('A =')
    print (to_value(a))
    print ('B =')
    print (to_value(b))
    assert np.allclose(to_value(a), to_value(b), atol=1e-4, rtol=1e-3)

def test_sgd():
    node = Variable(np.array(np.random.rand(3,3), dtype=precision))
    grad = Variable(np.array(np.random.rand(3,3), dtype=precision))
    opt = Sgd()

    set_cuda_active(True)
    dy_gpu = opt(grad, node)
    assert isinstance(dy_gpu, GPUValue)
    dy_gpu = Node(dy_gpu)
    dy_gpu.to_cpu()
    opt.reset()

    set_cuda_active(False)
    dy_cpu = opt(grad, node)
    assert isinstance(dy_cpu, Node)

    close(dy_gpu, dy_cpu)
