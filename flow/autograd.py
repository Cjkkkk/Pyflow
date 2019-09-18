import numpy as np

class Operation:
    def __init__(self):
        pass
    
    def __call__(self):
        return self.f()
    
    def f(self):
        raise NotImplementedError("should implement forward method")

class AddOperation(Operation):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def f(self):
        self.b.grad = self.output.grad * 1
        self.a.grad = self.output.grad * 1
        self.a.backward()
        self.b.backward()

class MulOperation(Operation):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def f(self):
        self.b.grad = self.output.grad * self.a.data
        self.a.grad = self.output.grad * self.b.data
        self.a.backward()
        self.b.backward()

class MMOperation(Operation):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def f(self):
        self.b.grad = np.matmul(np.transpose(self.a.data), self.output.grad)
        self.a.grad = np.matmul(self.output.grad, np.transpose(self.b.data))
        self.a.backward()
        self.b.backward()

class SumOperation(Operation):
    def __init__(self, output, a):
        super().__init__()
        self.a = a
        self.output = output
    
    def f(self):
        self.a.grad = np.ones(self.a.data.shape)
        self.a.backward()

class SquareLossOperation(Operation):
    def __init__(self, output, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.output = output
    
    def f(self):
        self.a.grad = 2.0 * self.a.data
        self.b.grad = -2.0 * self.b.data
        self.a.backward()
        self.b.backward()