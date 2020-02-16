from . import autograd
from .tensor import Tensor
import numpy as np


class Add(autograd.Function):  
    @staticmethod      
    def forward(ctx, a, b):
        new_tensor = Tensor(a.data + b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * 1
        a_grad = grad_output * 1
        return a_grad, b_grad

class Mul(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(a.data * b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        b_grad = grad_output * a.data
        a_grad = grad_output * b.data
        return a_grad, b_grad

class Sub(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        new_tensor = Tensor(a.data - b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * (-1)
        a_grad = grad_output * 1
        return a_grad, b_grad

class Truediv(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(a.data / b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        b_grad = grad_output * (-a.data) / (b.data ** 2)
        a_grad = grad_output / b.data
        return a_grad, b_grad

class MM(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(np.matmul(a.data, b.data))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        a_grad = np.matmul(grad_output, np.transpose(b.data))
        b_grad = np.matmul(np.transpose(a.data), grad_output)
        return a_grad, b_grad

class ReLU(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        copy = np.copy(a.data)
        copy[copy < 0] = 0
        return Tensor(copy)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors()[0]
        a_grad = np.copy(grad_output)
        a_grad[a.data < 0] = 0
        return a_grad

class Sum(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        new_tensor = Tensor(np.sum(a.data))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors()[0]
        a_grad = np.ones(a.data.shape)
        return a_grad

class SquareLoss(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(np.sum(np.square(a.data - b.data)))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        a_grad = 2.0 * (a.data - b.data)
        b_grad = -2.0 * (a.data - b.data)
        return a_grad, b_grad

add = Add.apply
mul = Mul.apply
sub = Sub.apply
true_div = Truediv.apply
mm = MM.apply
sum_ = Sum.apply
square_loss = SquareLoss.apply
relu = ReLU.apply