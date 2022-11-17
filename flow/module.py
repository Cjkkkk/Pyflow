from .tensor import Tensor
from .init import kaiming_uniform
from . import function as F
from .utils import _make_pair
import numpy as np
import math


class Module:
    def __init__(self):
        self._modules = dict()
        self._parameters = dict()
        self._training = True

    def train(self, mode=True):
        for module in self.modules():
            module._training = mode
    
    def eval(self):
        self.train(False)
    
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

    def modules(self):
        modules_list = [self]
        for module in modules_list:
            for name in module._modules:
                modules_list.append(module._modules[name])
        return modules_list

    def named_modules(self):
        modules_list = [("", self)]
        for prefix, module in modules_list:
            for name in module._modules:
                modules_list.append((prefix + "." + name, module._modules[name]))
        return modules_list

    def parameters(self):
        param = []
        for module in self.modules():
            param.extend(module._parameters.values())
        return param

    def named_parameters(self):
        params_list = []
        for module_name, module in self.named_modules():
            for param in module._parameters:
                params_list.append((module_name + "." + param, module._parameters[param]))
        return params_list
    
    def state_dict(self):
        dic = {}
        for module_name, module in self.named_modules():
            for para_name in module._parameters:
                dic[module_name + "." + para_name] = module._parameters[para_name]
        return dic
    
    def load_state_dict(self, state_dict):
        for module in self.modules():
            module._parameters = dict() # reset parameters, otherwise old parameters will remain
        
        for name in state_dict:
            s = name.split(".")
            module_path, para_name = s[1:-1], s[-1]
            module = self
            for p in module_path:
                module = getattr(module, p)
            setattr(module, para_name, state_dict[name])

    def to(self, device):
        raise NotImplemented()


class Identity(Module):
    def __init__(self, tensor):
        super().__init__()
        self.data = tensor

    def forward(self):
        return self.data

class Linear(Module):
    def __init__(self, input_channel, output_channel, bias=True):
        super().__init__()
        self.input_channel = input_channel
        self.outputchannel = output_channel
        # self.weight = Tensor(np.random.randn(input_channel, output_channel), require_grad=True)
        self.weight = Tensor(kaiming_uniform((input_channel, output_channel)), require_grad=True)
        if bias:
            # self.bias = Tensor(np.random.randn(output_channel), require_grad=True)
            self.bias = Tensor(kaiming_uniform((1, output_channel)), require_grad=True)
        else:
            self.bias = None
            
    def forward(self, x):
        y = F.mm(x, self.weight, self.bias)
        return y

class Conv2d(Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = _make_pair(kernel_size)
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)

        weight_scale = math.sqrt(self.kernel_size[0] * self.kernel_size[1] * self.input_channel / 2)
        if bias:
            self.bias = Tensor(np.random.standard_normal(self.output_channel) / weight_scale, require_grad=True)
        else:
            self.bias = None
        self.weight = Tensor(np.random.standard_normal(
            (self.output_channel, self.input_channel, self.kernel_size[0], self.kernel_size[1])) / weight_scale, require_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)