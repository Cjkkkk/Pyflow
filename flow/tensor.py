import numpy as np


class Tensor:
    def __init__(self, data, require_grad=False, is_leaf=True):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        self.data = data
        self.grad = None
        self.grad_fn = None
        self.is_leaf = is_leaf
        self.require_grad = require_grad
    
    def __add__(self, other):
        from . import function as F
        return F.add(self, other)

    def __mul__(self, other):
        from . import function as F
        return F.mul(self, other)

    def backward(self):
        if self.require_grad:
            if self.grad is None:
                self.grad = np.ones(self.data.shape)
            if not self.is_leaf:
                self.grad_fn()