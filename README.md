A simple Pytorch reimplementation.

[![CircleCI](https://circleci.com/gh/Cjkkkk/Pyflow.svg?style=svg)](https://circleci.com/gh/Cjkkkk/Pyflow)

## install/develop
### develop
```bash
python setup.py develop
```
## example

```python
from flow.module import Module, Linear
from flow.optim import SGD
from flow import function as F
from flow.tensor import Tensor, randn

class TwoFc(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 10)
        self.fc2 = Linear(10, 1)

    def forward(self, a):
        x = self.fc1(a)
        y = self.fc2(x)
        return y

# y = 3x_1 + 2x_2
model = TwoFc()
optim = SGD(model.parameters(), lr = 0.01)
for i in range(100):
    input = randn((1, 2))
    output = model(input)
    target = 3 * input[0, 0] + 2 * input[0, 1]
    loss = F.square_loss(output, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print("loss: ", loss)
```
## test
```bash
python test/test_gradient.py
```

PS: `pytorch` is required to check the correctness of implementation.

## todo
### Operator
* mm [done]
* relu [done]
* maxpool2d [done]
* conv2d [done]
* log_softmax [done]
* view [done]
* nll_loss [done]

### Autograd
* gradient accum in backward [done]
* gradient_check_tool [done]
* no_grad [done]
* grad as tensor [done]

### Module
* module load/store [done]

### Test
* test []

### Example
* mnist example []

### Memory Optimization
* ref count []
* in-place []
* normal sharing []

### Other
* dataloader []
* hide numpy from user []
* dot graph []