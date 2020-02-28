from flow.module import Module, Linear
from flow.optim import SGD
from flow import function as F
from flow.tensor import Tensor
import numpy as np

class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 10)
        self.fc2 = Linear(10, 1)

    def forward(self, a):
        x = self.fc1(a)
        y = F.relu(x)
        z = self.fc2(y)
        return z

# y = 3x_1^2 + 2x_2
model = Net()
optim = SGD(model.parameters(), lr = 0.005)
for i in range(100):
    input = Tensor(np.random.randn(1, 2))
    output = model(input)
    target = 3 * input.data[0, 0] * input.data[0, 0] + 2 * input.data[0, 1]
    loss = F.square_loss(output, Tensor(np.array([[target]])))
    loss.backward()
    optim.step()
    optim.zero_grad()
    print("loss", loss.data)