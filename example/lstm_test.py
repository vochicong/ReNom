import renom as rm
import numpy as np

x = rm.Variable(np.random.rand(1, 5))
y = np.random.rand(1, 1)

model = rm.Lstm(1)

eps = np.sqrt(np.finfo(np.float32).eps)

def auto_diff(function, node, *args):
    loss = function(*args)
    return loss.grad().get(node)


def numeric_diff(function, node, *args):
    shape = node.shape
    diff = np.zeros_like(node)

    if True:  # 5 point numerical diff
        coefficients1 = [1, -8, 8, -1]
        coefficients2 = [-2, -1, 1, 2]
        c = 12
    else:    # 3 point numerical diff
        coefficients1 = [1, -1]
        coefficients2 = [1, -1]
        c = 2

    node.to_cpu()
    node.setflags(write=True)
    for nindex in np.ndindex(shape):
        loss = 0
        for i in range(len(coefficients1)):
            dx = eps * node[nindex] if node[nindex] != 0 else eps
            node[nindex] += coefficients2[i] * dx
            node.to_cpu()
            ret_list = function(*args) * coefficients1[i]
            ret_list.to_cpu()
            node[nindex] -= coefficients2[i] * dx
            loss += ret_list
            loss.to_cpu()

        v = loss / (dx * c)
        v.to_cpu()
        diff[nindex] = v
    diff.to_cpu()
    return diff


def func(x):
    z = 0
    for i in range(10):
        z = rm.sum(model(x))
    model.truncate()
    return z

print(auto_diff(func, x, x))
print(numeric_diff(func, x, x))
print(np.allclose(auto_diff(func, x, x), numeric_diff(func, x, x)))
print(np.allclose(auto_diff(func, model.params.w, x), numeric_diff(func, model.params.w, x)))
print(np.allclose(auto_diff(func, model.params.b, x), numeric_diff(func, model.params.b, x)))
print(np.allclose(auto_diff(func, model.params.wr, x), numeric_diff(func, model.params.wr, x)))

