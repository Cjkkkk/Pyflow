from .tensor import Tensor
from .autograd import no_grad
import numpy as np


class optim:
    def __init__(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = None
    
    @no_grad()
    def step(self):
        raise NotImplementedError("should implement step method.")

class SGD(optim):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__()
        self.params = params
        self.u = {}
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    @no_grad()
    def step(self):
        for key, param in enumerate(self.params):
            if key not in self.u:
                self.u[key] = (1 - self.momentum) * (param.grad + param * self.weight_decay)
            else:
                self.u[key] = self.momentum * self.u[key] + (1 - self.momentum) * (param.grad + param * self.weight_decay)
            param -= self.u[key] * self.lr
            
            
class Adam(optim):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0):
        super().__init__()
        self.params = params
        self.u = {}
        self.v = {}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        
    @no_grad()
    def step(self):
        self.t += 1
        for key, param in enumerate(self.params):
            if key not in self.u:
                self.u[key] = (1 - self.beta1) * (param.grad + param * self.weight_decay)
            else:
                self.u[key] = self.beta1 * self.u[key] + (1 - self.beta1) * (param.grad + param * self.weight_decay)

            if key not in self.v:
                self.v[key] = (1 - self.beta2) * (param.grad + param * self.weight_decay) ** 2
            else:
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param.grad + param * self.weight_decay) ** 2

            # unbiased u v
            corrected_u = self.u[key] / (1 - self.beta1 ** self.t)
            corrected_v = self.v[key] / (1 - self.beta2 ** self.t)
            param -= corrected_u / (np.sqrt(corrected_v.data) + self.eps) * self.lr