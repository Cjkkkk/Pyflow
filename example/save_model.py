import pickle
from flow.module import Module, Linear
from flow.optim import SGD
from flow import function as F
from flow.tensor import Tensor
import numpy as np

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
    input = Tensor(np.random.randn(1, 2))
    output = model(input)
    target = 3 * input.data[0, 0] + 2 * input.data[0, 1]
    loss = F.square_loss(output, Tensor(np.array([[target]])))
    loss.backward()
    optim.step()
    optim.zero_grad()

with open("checkpoint.pth", "wb") as f:
    pickle.dump(model.state_dict(), f)

with open("checkpoint.pth", "rb") as f:
    state_dict = pickle.load(f)

model = TwoFc()
model.load_state_dict(state_dict)
input = Tensor(np.random.randn(1, 2))
output = model(input)
target = 3 * input.data[0, 0] + 2 * input.data[0, 1]
loss = F.square_loss(output, Tensor(np.array([[target]])))
print("loss", loss.data)