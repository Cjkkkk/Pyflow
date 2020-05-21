import numpy as np
import traceback
import functools
from .tensor import Tensor, ones


class no_grad:
    _is_grad_enabled = True
    def __call__(self, func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrap

    def __enter__(self):
        self.prev = no_grad._is_grad_enabled
        no_grad._is_grad_enabled = False

    def __exit__(self, exc_type, exc_value, tb):
        # TODO how to correctly write a exit func?
        # if exc_type is not None:
        #     traceback.print_exception(exc_type, exc_value, tb)
        no_grad._is_grad_enabled = self.prev
        return False

 
class Context:
    def __init__(self):
        self.store = None
    
    def record_version(self):
        if self.store is not None:
            self.record_version = [v.version for v in self.store if isinstance(v, Tensor)]
    
    def save_for_backward(self, *values):
        self.store = values
        # update ref_count for memory optimization
        for v in self.store:
            if isinstance(v, Tensor):
                # TODO raise error if variable needed by gradient computation is used in inplace operation
                v.ref_count += 1
    @property
    def saved_tensors(self):
        current_version = [v.version for v in self.store if isinstance(v, Tensor)]
        for idx, version in enumerate(current_version):
            if (version != self.record_version[idx]):
                tensor = [v for v in self.store if isinstance(v, Tensor)][idx]
                raise RuntimeError("one of the tensors needed for gradient is being used in gradient computation. Tensor is output of %s" % str(type(tensor.grad_fn)))
        return self.store

def register_backward(func, output):
    if no_grad._is_grad_enabled:
        require_grad = False
        for i in func.inputs: # if all input does not require grad, output does not require grad
            require_grad = i.require_grad or require_grad
            output.version += i is output
        
        if output.require_grad and output.is_leaf and output.version != 0:
            raise RuntimeError("leaf tensor with require_grad=True can not be used in inplace operation.")
        
        if require_grad:
            func.ctx.record_version()
            output.grad_fn = func
            output.is_leaf = False
            output.require_grad = require_grad

    # TODO inplace operation should no overwrite its grad_fn, how to record multiple grad_fn?
    # else:
    #     # just to make sure no state is modified
    #     output.grad_fn = None
    #     output.is_leaf = True
    #     output.require_grad = False

class Function:
    def __init__(self, *args, **kwargs):
        self.ctx = Context()
        self.inputs = None
        self.next_functions = None
        if no_grad._is_grad_enabled:
            self.inputs = [ v for v in [*args, *kwargs.values()] if isinstance(v, Tensor) ]
            self.next_functions = [inp.grad_fn for inp in self.inputs if inp.require_grad]
            for inp in self.inputs:
                # update ref_count for memory optimization
                inp.ref_count += 1
    
    @classmethod
    def apply(cls, *args, **kwargs):
        func = cls(*args, **kwargs)
        with no_grad():
            # should not build computation graph in function forward method
            output = cls.forward(func.ctx, *args, **kwargs)
        register_backward(func, output)
        return output

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("should implement forward method")
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("should implement backward method")

def backward(tensor, grad_fn, grad=None):
    # TODO why can not add no_grad decorator to backward?
    if tensor.require_grad:
        if grad is None:
            grad = ones(tensor.data.shape)
        if tensor.grad is None or tensor.version != 0:
            tensor.grad = grad
        else:
            # TODO fix this bug, iadd is not implemented
            tensor.grad += grad
        if not tensor.is_leaf:
            input_grads = grad_fn.backward(grad_fn.ctx, tensor.grad)  
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            for idx, next_fn in enumerate(grad_fn.next_functions):
                backward(grad_fn.inputs[idx], next_fn, input_grads[idx])