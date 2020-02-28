import flow.function as F
from flow.tensor import Tensor
from flow.module import Conv2d, Linear, Module
import numpy as np


class Net(Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(1, 20, 5, 1)
        self.conv2 = Conv2d(20, 50, 5, 1)
        self.fc1 = Linear(4*4*50, 500)
        self.fc2 = Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.maxpool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.maxpool2d(x, 2, 2)
        x = F.view(x, (-1, 4*4*50))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

print(Tensor([1,2,3,4]), Tensor([1,2,3]).shape())