A simple Pytorch reimplementation.

## install
```bash
pip install py-flow
```

## example

```python
from flow.module import Module, IdentityLayer
from flow.optim import SGD
from flow import function as F
from flow.tensor import Tensor


class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.a = IdentityLayer(Tensor([[4.0, 5.0]], require_grad=True))
        self.b = IdentityLayer(Tensor([[5.0], [6.0]], require_grad=True))
        self.c = IdentityLayer(Tensor([[1.0,2.0], [3.0,4.0]], require_grad=True))

    def forward(self):
        x = F.mm(self.b(), self.a())
        y = self.c() + x
        z = F.sum_(y)
        return z

net = MyNet()
optim = SGD(net.parameters(), lr = 0.001)

output = net()
output.backward()

optim.step()
print(output.data, output.grad, net.a.data.grad, net.a.data.data)
```

## todo
* gradient accum in backward when input is consumed by multi operators
