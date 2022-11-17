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
        self.store = []
    
    def record_version(self):
        self.record_version = [v.version for v in self.store if isinstance(v, Tensor)]
    
    def save_for_backward(self, *values):
        self.store = values
        # update ref_count for memory optimization
        for v in self.store:
            if isinstance(v, Tensor):
                v.backward_ref_count += 1
    
    @property
    def saved_tensors(self):
        current_version = [v.version for v in self.store if isinstance(v, Tensor)]
        for idx, version in enumerate(current_version):
            if (version != self.record_version[idx]):
                tensor = [v for v in self.store if isinstance(v, Tensor)][idx]
                raise RuntimeError("one of the tensors needed for gradient is being used in gradient computation. Tensor is output of %s" % str(type(tensor.grad_fn)))
        return self.store

def register_backward(grad_fn, output):
    if no_grad._is_grad_enabled:
        require_grad = False
        for i in grad_fn.fn.inputs: # if all input does not require grad, output does not require grad
            require_grad = i.require_grad or require_grad
            output.version += i is output
        
        if output.require_grad and output.is_leaf and output.version != 0:
            raise RuntimeError("leaf tensor with require_grad=True can not be used in inplace operation.")
        
        if require_grad:
            grad_fn.record_version()
            output.grad_fn = grad_fn
            output.is_leaf = False
            output.require_grad = require_grad

    # TODO inplace operation should no overwrite its grad_fn, how to record multiple grad_fn?
    # else:
    #     # just to make sure no state is modified
    #     output.grad_fn = None
    #     output.is_leaf = True
    #     output.require_grad = False

class BackwardFunction(Context):
    def __call__(self, *args, **kwargs):
        return self._forward_cls.backward(self, *args, **kwargs)

class FunctionMeta(type):
    def __new__(cls, name, bases, attrs):
        forward_cls = super().__new__(cls, name, bases, attrs)
        forward_cls._backward_cls = type(name + "backward", (BackwardFunction, ), {"_forward_cls": forward_cls})
        return forward_cls

class Function(metaclass=FunctionMeta):
    def __new__(cls, *args, **kwargs):
        forward_function = object.__new__(cls)
        backward_function = cls._backward_cls()
        forward_function.grad_fn = backward_function
        backward_function.fn = forward_function
        return forward_function

    def __init__(self, *args, **kwargs):
        self.inputs = None
        self.grad_fn.next_functions = None
        
        if no_grad._is_grad_enabled:
            self.inputs = [ v for v in [*args, *kwargs.values()] if isinstance(v, Tensor) ]
            self.grad_fn.next_functions = [inp.grad_fn for inp in self.inputs]
            for inp in self.inputs:
                # update ref_count for memory optimization
                inp.forward_ref_count += 1
    
    def __call__(self, *args, **kwargs):
        return self.forward(self.grad_fn, *args, **kwargs)
    
    @classmethod
    def apply(cls, *args, **kwargs):
        func = cls(*args, **kwargs)
        with no_grad():
            # should not build computation graph in function forward method
            output = func(*args, **kwargs)
        register_backward(func.grad_fn, output)
        return output

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("should implement forward method")
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("should implement backward method")

def update_ref_count(tensor, ctx):
    tensor.forward_ref_count -= 1
    for t in ctx.saved_tensors:
        if tensor is t:
            tensor.backward_ref_count -= 1

def backward(tensor, grad_fn, grad=None):
    # TODO why can not add no_grad decorator to backward?
    if tensor.require_grad:
        if grad is None:
            grad = ones(tensor.shape)
        if tensor.grad is None:
            tensor.grad = grad
        else:
            # TODO fix this bug, iadd is not implemented
            tensor.grad += grad
        if not tensor.is_leaf and tensor.forward_ref_count == 0:
            input_grads = grad_fn(tensor.grad)
            input_grads = (input_grads,) if not isinstance(input_grads, tuple) else input_grads
            for idx, next_fn in enumerate(grad_fn.next_functions):
                update_ref_count(grad_fn.fn.inputs[idx], grad_fn)
                backward(grad_fn.fn.inputs[idx], next_fn, input_grads[idx])