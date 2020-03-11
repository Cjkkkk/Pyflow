import flow.function as F
from flow.tensor import Tensor
from flow.utils import gradient_check
from flow.module import MaxPool2d, Conv2d
import numpy as np
import torch

# check using gradient_check tools
def check_sum_():
    gradient_check(F.sum_, Tensor([[1,2,3,4], [5, 6, 7, 8]], require_grad=True))

def check_nll_loss():
    gradient_check(F.nll_loss, Tensor([[1, 2, 3, 4], [5, 6, 7, 8]], require_grad=True), Tensor(np.array([0, 1])), True)


# check using pytorch
def check_conv2d():
    torch_conv2d = torch.nn.Conv2d(3, 2, 2, bias=False)
    torch_input = torch.rand((1, 3, 3, 3), requires_grad=True)
    torch_output = torch_conv2d(torch_input)
    torch_output.backward(torch.ones(torch_output.shape))

    flow_conv2d = Conv2d(3, 2, 2, bias=False)
    flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
    flow_conv2d.weight = Tensor(torch_conv2d.weight.detach().numpy(), require_grad=True)
    flow_output = flow_conv2d(flow_input)
    flow_output.backward()

    assert np.allclose(torch_output.detach().numpy(), flow_output.data)
    assert np.allclose(torch_conv2d.weight.grad.detach().numpy(), flow_conv2d.weight.grad)
    assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad)

def check_maxpool2d():
    torch_maxpool2d = torch.nn.MaxPool2d(2, stride=1)
    torch_input = torch.rand((1, 3, 3, 3), requires_grad=True)
    torch_output = torch_maxpool2d(torch_input)
    torch_output.backward(torch.ones(torch_output.shape))

    flow_maxpool2d = MaxPool2d(2)
    flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
    flow_output = flow_maxpool2d(flow_input)
    flow_output.backward()

    assert np.allclose(torch_output.detach().numpy(), flow_output.data)
    assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad)

if __name__ == '__main__':
    check_sum_()
    check_nll_loss()
    check_conv2d()
    check_maxpool2d()