import flow.function as F
from flow.tensor import Tensor
from flow.module import Conv2dLayer
import numpy as np

input = Tensor(np.random.randint(5, size=(1, 3, 3, 3)))
conv2d = Conv2dLayer(3, 2, 2, stride=1, padding=0, bias=True)
result = conv2d(input)
result.backward()