from .tensor import Tensor

class optim:
    def __init__(self):
        pass
    def step(self, params):
        raise NotImplementedError("should implement step method.")

class SGD(optim):
    def __init__(self, params, lr):
        super().__init__()
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad