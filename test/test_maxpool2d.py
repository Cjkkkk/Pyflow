import flow.function as F
from flow.tensor import Tensor
from flow.module import MaxPool2d
import numpy as np


input = Tensor(np.random.randint(5, size=(1, 2, 4, 4)), require_grad=True)
pool2d = MaxPool2d(2, stride=2, padding=1)
result = pool2d(input)
result.backward()