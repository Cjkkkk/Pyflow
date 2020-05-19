import flow
from flow.module import Module, Linear
from flow.optim import SGD
from flow import function as F
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
    input = flow.randn((1, 2))
    output = model(input)
    target = 3 * input[0, 0] * input[0, 0] + 2 * input[0, 1]
    loss = F.square_loss(output, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print("loss", loss)