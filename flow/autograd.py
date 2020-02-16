import numpy as np
from .tensor import Tensor

class Context:
    def save_for_backward(self, *tensors):
        self.store = tensors
    def saved_tensors(self):
        return self.store

class Function:
    def __init__(self):
        self.ctx = Context()
        self.inputs = None

    def register_backward(self, output):
        output.grad_fn = self
        output.is_leaf = False
        for i in self.inputs: # if all input does not require grad, output does not require grad
            if isinstance(i, Tensor):
                output.require_grad = i.require_grad or output.require_grad
    
    @classmethod
    def apply(cls, *args):
        f = cls()
        f.inputs = args
        output = cls.forward(f.ctx, *args)
        f.register_backward(output)
        return output

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("should implement forward method")
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("should implement backward method")

def backward(tensor, grad):
    if tensor.require_grad:
        if grad is None:
            grad = np.ones(tensor.data.shape)
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad
        if not tensor.is_leaf:
            input_grads = tensor.grad_fn.backward(tensor.grad_fn.ctx, tensor.grad)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            for idx, input in enumerate(tensor.grad_fn.inputs):
                if isinstance(input, Tensor):
                    backward(input, input_grads[idx])