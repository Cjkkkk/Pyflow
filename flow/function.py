from . import autograd
from .tensor import Tensor
import numpy as np


def register_backward_hook(output, function, *input):
    output.grad_fn = function(output, *input)
    output.is_leaf = False
    for i in input: # if all input does not require grad, output does not require grad
        output.require_grad = i.require_grad or output.require_grad

def add(a, b):
    new_tensor = Tensor(a.data + b.data)
    register_backward_hook(new_tensor, autograd.AddFunction, a, b)
    return new_tensor

def mul(a, b):
    new_tensor = Tensor(a.data * b.data)
    register_backward_hook(new_tensor, autograd.MulFunction, a, b)
    return new_tensor

def sub(a, b):
    new_tensor = Tensor(a.data - b.data)
    register_backward_hook(new_tensor, autograd.SubFunction, a, b)
    return new_tensor

def truediv(a, b):
    new_tensor = Tensor(a.data / b.data)
    register_backward_hook(new_tensor, autograd.TruedivFunction, a, b)
    return new_tensor

def pow(a, b):
    new_tensor = Tensor(a.data ** b.data)
    register_backward_hook(new_tensor, autograd.PowFunction, a, b)
    return new_tensor

def mm(a, b):
    new_tensor = Tensor(np.matmul(a.data, b.data))
    register_backward_hook(new_tensor, autograd.MMFunction, a, b)
    return new_tensor

def sum_(a):
    new_tensor = Tensor(np.sum(a.data))
    register_backward_hook(new_tensor, autograd.SumFunction, a)
    return new_tensor

def square_loss(a, b):
    new_tensor = Tensor(np.sum(np.square(a.data - b.data)))
    register_backward_hook(new_tensor, autograd.SquareLossFunction, a, b)
    return new_tensor