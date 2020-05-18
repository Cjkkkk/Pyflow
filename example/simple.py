from flow.module import Module, Identity
from flow.optim import SGD
from flow import function as F
from flow.tensor import Tensor


class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.a = Identity(Tensor([[4.0, 5.0]], require_grad=True))
        self.b = Identity(Tensor([[5.0], [6.0]], require_grad=True))
        self.c = Identity(Tensor([[1.0,2.0], [3.0,4.0]], require_grad=True))

    def forward(self):
        x = F.mm(self.b(), self.a())
        y = self.c() + x
        z = F.sum(y)
        return z

net = MyNet()
optim = SGD(net.parameters(), lr = 0.001)

output = net()
output.backward()

optim.step()
# 109.0 1.0 [[11. 11.]] [[4.011 5.011]]
print(output.data, output.grad, net.a.data.grad, net.a.data.data)