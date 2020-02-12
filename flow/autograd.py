import numpy as np
from .tensor import Tensor

class Function:
    def __init__(self, *args):
        self.inputs = args
    
    def __call__(self):
        return self.backward()
    
    def backward(self):
        raise NotImplementedError("should implement backward method")

class AddFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        b_grad = grad_output * 1
        a_grad = grad_output * 1
        return a_grad, b_grad

class MulFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        b_grad = grad_output * self.a.data
        a_grad = grad_output * self.b.data
        return a_grad, b_grad

class SubFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        b_grad = grad_output * (-1)
        a_grad = grad_output * 1
        return a_grad, b_grad

class TruedivFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        b_grad = grad_output * (-self.a.data) / (self.b.data ** 2)
        a_grad = grad_output / self.b.data
        return a_grad, b_grad

class MMFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        b_grad = np.matmul(np.transpose(self.a.data), grad_output)
        a_grad = np.matmul(grad_output, np.transpose(self.b.data))
        return a_grad, b_grad

class SumFunction(Function):
    def __init__(self, a):
        super().__init__(*args)
    
    def backward(self, grad_output):
        a_grad = np.ones(self.a.data.shape)
        return a_grad

class SquareLossFunction(Function):
    def __init__(self, *args):
        super().__init__(*args)
    
    def backward(self, grad_output):
        a_grad = 2.0 * self.a.data
        b_grad = -2.0 * self.b.data
        return a_grad, b_grad
