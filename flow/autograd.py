import numpy as np
from .tensor import Tensor, ones


class no_grad:
    _is_grad_enabled = True
    def __enter__(self):
        self.prev = no_grad._is_grad_enabled
        no_grad._is_grad_enabled = False

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        no_grad._is_grad_enabled = self.prev

class Context:
    def save_for_backward(self, *tensors):
        self.store = tensors
        # update ref_count for memory optimization
        for s in self.store:
            if isinstance(s, Tensor):
                s.ref_count += 1
    
    def saved_tensors(self):
        return self.store

def register_backward(func, output):
    if no_grad._is_grad_enabled:
        for i in func.inputs: # if all input does not require grad, output does not require grad
            if isinstance(i, Tensor):
                output.require_grad = i.require_grad or output.require_grad
        if output.require_grad:
            output.grad_fn = func
            output.is_leaf = False

    else:
        # just to make sure no state is modified
        output.grad_fn = None
        output.is_leaf = True
        output.require_grad = False

class Function:
    def __init__(self, *args, **kwargs):
        self.ctx = Context()
        self.inputs = [ v for v in [*args, *kwargs.values()] if isinstance(v, Tensor) ]
        for inp in self.inputs:
            # update ref_count for memory optimization
            inp.ref_count += 1
    
    @classmethod
    def apply(cls, *args, **kwargs):
        func = cls(*args)
        output = cls.forward(func.ctx, *args, **kwargs)
        register_backward(func, output)
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
            grad = ones(tensor.data.shape)
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