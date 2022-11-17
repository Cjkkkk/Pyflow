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
        self.forward_ref_count = 0
        self.backward_ref_count = 0
        self.version = 0

    @property
    def require_grad(self):
        return self._require_grad
    
    @require_grad.setter
    def require_grad(self, new):
        if self.data.dtype != np.float64 and self.data.dtype != np.float32 and new:
            raise RuntimeError("only Tensors of floating point type can set requires_grad=True")
        self._require_grad = new
    
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
    
    def __iadd__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.add(self, other, inplace=True)
    
    def __isub__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.sub(self, other, inplace=True)
    
    def __imul__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.mul(self, other, inplace=True)
    
    def __itruediv__(self, other):
        from . import function as F
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return F.true_div(self, other, inplace=True)
    
    def __floordiv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_tensor = Tensor(self.data // other.data)
        return new_tensor
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # TODO should have gradient version
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
    
    def reshape(self, *new_shape):
        self.data = self.data.reshape(*new_shape)
        return self

    def argmax(self, *args, **kwargs):
        return Tensor(self.data.argmax(*args, **kwargs))

    def min(self, *args, **kwargs):
        from . import function as F
        return F.min(self)

    def max(self, *args, **kwargs):
        from . import function as F
        return F.max(self)
    
    def sum(self, *args, **kwargs):
        from . import function as F
        return F.sum(self)

    def copy(self):
        new_tensor = Tensor(np.copy(self.data))
        return new_tensor

    def astype(self, new_type):
        return Tensor(self.data.astype(new_type))
    
    def item(self):
        if self.size() != 1:
            raise ValueError("tensor size is larger than 1, ambiguous value.")
        return self.data.item(0)
    
    def _print_graph(self, id=0, ident=""):
        pass

    @property
    def shape(self):
        return self.data.shape
    
    # Note: numpy size returns total amount but pytorch size() is alias of shape
    def size(self):
        return self.data.size
    
    def dim(self):
        return len(self.shape)
    
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
        with autograd.no_grad():
            # should not build computation graph in backward method
            autograd.backward(self, self.grad_fn, grad)
    
    def __getitem__(self, key):
        if isinstance(key, Tensor):
            # assum key is mask
            return Tensor(self.data[key.data])
        else:
            return Tensor(self.data[key])

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data

        if isinstance(key, Tensor):
            # assum key is mask
            self.data[key.data] = value
        else:
            self.data[key] = value

def transpose(tensor):
    # TODO: fix shape here
    tensor.data = np.transpose(tensor.data)
    return tensor

def empty(shape, require_grad=False):
    return Tensor(np.empty(shape), require_grad)

def ones(shape, require_grad=False):
    return Tensor(np.ones(shape), require_grad)

def zeros(shape, require_grad=False):
    return Tensor(np.zeros(shape), require_grad)

def randn(shape, require_grad=False):
    return Tensor(np.random.randn(*shape), require_grad)

def rand(shape, require_grad=False):
    return Tensor(np.random.rand(*shape), require_grad)

def stack(tensors):
    return Tensor(np.stack([t.data for t in tensors]))