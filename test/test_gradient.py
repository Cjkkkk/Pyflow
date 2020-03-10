import flow.function as F
from flow.tensor import Tensor
from flow.utils import gradient_check
from flow.module import MaxPool2d, Conv2d
import numpy as np


gradient_check(F.sum_, Tensor([[1,2,3,4], [5, 6, 7, 8]], require_grad=True))
gradient_check(F.nll_loss, Tensor([[1, 2, 3, 4], [5, 6, 7, 8]], require_grad=True), Tensor(np.array([0, 1])), True)

# input = Tensor(np.random.randint(5, size=(1, 2, 4, 4)), require_grad=True)
# pool2d = MaxPool2d(2, stride=2, padding=1)
# gradient_check(pool2d, input)

input = Tensor(np.random.randint(5, size=(1, 3, 3, 3)))
weight = Tensor(np.random.randint(5, size=(2, 3, 2, 2)), require_grad=True)
gradient_check(F.conv2d, input, weight, None, (1, 1), (0, 0))