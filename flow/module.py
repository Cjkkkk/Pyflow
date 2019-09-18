from .tensor import Tensor
from . import function as F
import numpy as np

class Module:
    def __init__(self):
        self._modules = dict()
        self._parameters = dict()

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            pass
        object.__setattr__(self, name, value)
    
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def __str__(self):
        return "Module " + str(self.data)

    def forward(self, *input):
        raise NotImplementedError("should implement forward method.")

    def parameters(self):
        param = []
        for name in self._parameters:
            param.append(self._parameters[name])
        for name in self._modules:
            param.extend(self._modules[name].parameters())
        return param

class IdentityLayer(Module):
    def __init__(self, tensor):
        super().__init__()
        self.data = tensor

    def forward(self):
        return self.data

class fullyConnectLayer(Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_channel = input_channel
        self.outputchannel = output_channel
        self.weight = Tensor(np.random.randn(input_channel, output_channel), require_grad=True)

    def forward(self, x):
        y = F.mm(x, self.weight)
        return y