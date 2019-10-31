import numpy as np
from .tensor import Tensor

class Function:
    def __init__(self):
        pass
    
    def __call__(self):
        return self.backward()
    
    def backward(self):
        raise NotImplementedError("should implement backward method")

class AddFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.b.grad = self.output.grad * 1
        self.a.grad = self.output.grad * 1
        self.a.backward()
        self.b.backward()

class MulFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.b.grad = self.output.grad * self.a.data
        self.a.grad = self.output.grad * self.b.data
        self.a.backward()
        self.b.backward()

class SubFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.b.grad = self.output.grad * (-1)
        self.a.grad = self.output.grad * 1
        self.a.backward()
        self.b.backward()

class TruedivFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.b.grad = self.output.grad * (-self.a.data) / (self.b.data ** 2)
        self.a.grad = self.output.grad / self.b.data
        self.a.backward()
        self.b.backward()

class PowFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        pass

class MMFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.b.grad = np.matmul(np.transpose(self.a.data), self.output.grad)
        self.a.grad = np.matmul(self.output.grad, np.transpose(self.b.data))
        self.a.backward()
        self.b.backward()

class SumFunction(Function):
    def __init__(self, output, a):
        super().__init__()
        self.a = a
        self.output = output
    
    def backward(self):
        self.a.grad = np.ones(self.a.data.shape)
        self.a.backward()

class SquareLossFunction(Function):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def backward(self):
        self.a.grad = 2.0 * self.a.data
        self.b.grad = -2.0 * self.b.data
        self.a.backward()
        self.b.backward()