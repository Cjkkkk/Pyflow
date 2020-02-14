import numpy as np
from .tensor import Tensor

class Function:
    def __init__(self):
        pass

    def __call__(self, *args):
        self.inputs = args
        output = self.forward(*args)
        self.register_backward_hook(output)
        return output

    def register_backward_hook(self, output):
        output.grad_fn = self
        output.is_leaf = False
        for i in self.inputs: # if all input does not require grad, output does not require grad
            output.require_grad = i.require_grad or output.require_grad
        
    def forward(self, *args):
        raise NotImplementedError("should implement forward method")
       
    def backward(self, grad_output):
        raise NotImplementedError("should implement backward method")
