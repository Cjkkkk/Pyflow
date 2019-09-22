from . import autograd
from .tensor import Tensor
import numpy as np


def register_backward_hook(output, operation, *input):
    output.grad_fn = operation(output, *input)
    output.is_leaf = False
    for i in input: # if all input does not require grad, output does not require grad
        output.require_grad = i.require_grad or output.require_grad

def add(a, b):
    new_tensor = Tensor(a.data + b.data)
    register_backward_hook(new_tensor, autograd.AddOperation, a, b)
    return new_tensor

def mul(a, b):
    new_tensor = Tensor(a.data * b.data)
    register_backward_hook(new_tensor, autograd.MulOperation, a, b)
    return new_tensor

def sub(a, b):
    new_tensor = Tensor(a.data - b.data)
    register_backward_hook(new_tensor, autograd.SubOperation, a, b)
    return new_tensor

def truediv(a, b):
    new_tensor = Tensor(a.data / b.data)
    register_backward_hook(new_tensor, autograd.TruedivOperation, a, b)
    return new_tensor

def pow(a, b):
    new_tensor = Tensor(a.data ** b.data)
    register_backward_hook(new_tensor, autograd.PowOperation, a, b)
    return new_tensor

def mm(a, b):
    new_tensor = Tensor(np.matmul(a.data, b.data))
    register_backward_hook(new_tensor, autograd.MMOperation, a, b)
    return new_tensor

def sum_(a):
    new_tensor = Tensor(np.sum(a.data))
    register_backward_hook(new_tensor, autograd.SumOperation, a)
    return new_tensor

def square_loss(a, b):
    new_tensor = Tensor(np.sum(np.square(a.data - b.data)))
    register_backward_hook(new_tensor, autograd.SquareLossOperation, a, b)
    return new_tensor