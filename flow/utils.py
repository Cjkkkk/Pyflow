import numpy as np
from .tensor import Tensor

def compute_loc(idx, shape):
    loc = [0] * len(shape)
    for i in range(len(shape) - 1):
        prod = np.prod(shape[i + 1:])
        loc[i] = idx // prod
        idx = idx % prod
    loc[-1] = idx % shape[-1]
    return tuple(loc)

def gradient_check(f, *args):
    out = f(*args)
    out.backward()
    eps = 1e-8
    for arg in args:
        # only check gradient of tensot
        if isinstance(arg, Tensor) and arg.require_grad:
            shape = arg.data.shape
            gradient = np.zeros(shape)
            size = np.size(arg.data)
            # traverse all elements
            for idx in range(size):
                loc = compute_loc(idx, shape)
                arg.data[loc] += eps
                out_eps = f(*args)
                gradient[loc] = np.sum(out_eps.data - out.data) / eps
                arg.data[loc] -= eps
            print(gradient, arg.grad)
            assert np.allclose(gradient, arg.grad)