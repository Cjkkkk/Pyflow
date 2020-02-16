from .tensor import Tensor
from . import function as F
import numpy as np
import math


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
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *inputs):
        raise NotImplementedError("should implement forward method.")

    def parameters(self):
        param = []
        for name in self._parameters:
            param.append(self._parameters[name])
        for name in self._modules:
            param.extend(self._modules[name].parameters())
        return param

    def to(self, device):
        for param in self.parameters():
            param.to(device)
        
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

class Conv2dLayer(Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        weight_scale = math.sqrt(kernel_size * kernel_size * self.input_channel / 2)
        if bias:
            self.bias = Tensor(np.random.standard_normal(self.output_channel) / weight_scale)
        else:
            self.bias = None
        self.weight = Tensor(np.random.standard_normal(
            (self.output_channel, self.input_channel, kernel_size, kernel_size)) / weight_scale)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)