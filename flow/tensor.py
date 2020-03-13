import numpy as np


class Tensor:
    def __init__(self, data, require_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.grad_fn = None
        # if required_grad is False, is_leaf is True
        # if required_grad is True and Tensor is created by user, is_leaf is True, False otherwise
        self.is_leaf = True
        self.require_grad = require_grad
    
    def __add__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.add(self, other)

    def __mul__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.mul(self, other)
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __sub__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.sub(self, other)
   
    def __truediv__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.true_div(self, other)
    
    def __floordiv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data // other.data)
        return new_tensor
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data ** other.data)
        return new_tensor
    
    def __mod__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data % other.data)
        return new_tensor

    def __lt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data < other.data)
        return new_tensor
    
    def __gt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data > other.data)
        return new_tensor

    def __le__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data <= other.data)
        return new_tensor

    def __ge__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data >= other.data)
        return new_tensor

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data == other.data)
        return new_tensor

    def __ne__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data != other.data)
        return new_tensor
    
    def __str__(self):
        return "tensor(%s)" % self.data
    
    def reshape(self, new_shape):
        self.data = self.data.reshape(new_shape)
        return self
    
    def copy(self):
        new_tensor = Tensor(np.copy(self.data), self.require_grad)
        return new_tensor
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def astype(self, type):
        self.data.astype(type)
        return self
    
    def to(self, device):
        # TODO use cupy to enable GPU usage
        raise NotImplementedError("to method is not supported yet.")
    
    def backward(self, grad=None):
        from . import autograd
        autograd.backward(self, grad)
    
    def __getitem__(self, key):
        return Tensor(self.data[key], self.require_grad)

    def __setitem__(self, key, value):
        self.data[key] = value

def ones(shape, require_grad=False):
    return Tensor(np.ones(shape), require_grad)

def randn(shape, require_grad=False):
    return Tensor(np.random.randn(*shape), require_grad)