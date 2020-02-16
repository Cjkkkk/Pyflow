from .tensor import Tensor

class optim:
    def __init__(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = None
    
    def step(self):
        raise NotImplementedError("should implement step method.")

class SGD(optim):
    def __init__(self, params, lr):
        super().__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad