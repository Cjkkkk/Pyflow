import numpy as np


class Tensor:
    def __init__(self, data, require_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        self.data = data
        self.grad = None
        self.grad_fn = None
        # if required_grad is False, is_leaf is True
        # if required_grad is True and Tensor is created by user, is_leaf is True, False otherwise
        self.is_leaf = True
        self.require_grad = require_grad
    
    def __add__(self, other):
        from . import function as F
        return F.Add()(self, other)

    def __mul__(self, other):
        from . import function as F
        return F.Mul()(self, other)
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __sub__(self, other):
        from . import function as F
        return F.Sub()()(self, other)
   
    def __truediv__(self, other):
        from . import function as F
        return F.TrueDiv()(self, other)
    
    def __floordiv__(self, other):
        new_tensor = Tensor(self.data // other.data)
        return new_tensor
    
#     def __pow__(self, other):
#         from . import function as F
#         return F.pow(self, other)
    
    def __mod__(self, other):
        new_tensor = Tensor(self.data % self.data)
        return new_tensor

    def __lt__(self, other):
        new_tensor = Tensor(self.data < other.data)
        return new_tensor
    
    def __gt__(self, other):
        new_tensor = Tensor(self.data > other.data)
        return new_tensor

    def __le__(self, other):
        new_tensor = Tensor(self.data <= other.data)
        return new_tensor

    def __ge__(self, other):
        new_tensor = Tensor(self.data >= other.data)
        return new_tensor

    def __eq__(self, other):
        new_tensor = Tensor(self.data == other.data)
        return new_tensor

    def __ne__(self, other):
        new_tensor = Tensor(self.data != other.data)
        return new_tensor

    def backward(self, grad=None):
        if self.require_grad:
            if grad is None:
                self.grad = np.ones(self.data.shape)
            else:
                self.grad = grad
            if not self.is_leaf:
                input_grads = self.grad_fn.backward(self.grad)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)
                for idx, input in enumerate(self.grad_fn.inputs):
                    input.backward(input_grads[idx])