import flow
import flow.function as F
from flow.module import Module, Linear
from flow.optim import SGD


class TwoFc(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 10)
        self.fc2 = Linear(10, 1)

    def forward(self, a):
        x = self.fc1(a)
        x = self.fc2(x)
        return x

# y = 3x_1 + 2x_2
model = TwoFc()
optim = SGD(model.parameters(), lr = 0.01)
for i in range(100):
    input = flow.randn((1, 2))
    output = model(input)
    target = 3 * input[0, 0] + 2 * input[0, 1]
    loss = F.square_loss(output, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    print("loss: ", loss)