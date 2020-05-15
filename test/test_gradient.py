import unittest
import numpy as np
import torch
import torch.nn.functional as PF

import flow.function as F
from flow.tensor import Tensor, randn
from flow.utils import gradient_check
from flow.module import MaxPool2d, Conv2d



class TestGradientAuto(unittest.TestCase):
    # check using gradient_check tools
    def test_sum_auto(self):
        gradient_check(F.sum_, randn((5,4), require_grad=True))

    def test_nll_loss_auto(self):
        gradient_check(F.nll_loss, randn((2,4), require_grad=True), Tensor([0, 1]))

    def test_square_loss_auto(self):
        gradient_check(F.square_loss, randn((2,4), require_grad=True), randn((2,4), require_grad=True))

    def test_logsoftmax_auto(self):
        gradient_check(F.log_softmax, randn((5,4), require_grad=True))
    
    def test_relu_auto(self):
        gradient_check(F.relu, randn((5,4), require_grad=True))

    def test_mm_auto(self):
        gradient_check(F.mm, randn((5,4), require_grad=True), randn((4,5), require_grad=True))
    
    def test_view_auto(self):
        gradient_check(F.view, randn((4,5), require_grad=True), (20, 1))

class TestGradientPytorch(unittest.TestCase):
    # check using pytorch
    def test_logsoftmax(self):
        torch_input = torch.rand((3, 2), requires_grad=True)
        torch_output = PF.log_softmax(torch_input, dim=1)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.log_softmax(flow_input)
        flow_output.backward()
        
        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

    def test_conv2d(self):
        torch_conv2d = torch.nn.Conv2d(3, 2, 2, bias=False)
        torch_input = torch.rand((1, 3, 3, 3), requires_grad=True)
        torch_output = torch_conv2d(torch_input)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_conv2d = Conv2d(3, 2, 2, bias=False)
        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_conv2d.weight = Tensor(torch_conv2d.weight.detach().numpy(), require_grad=True)
        flow_output = flow_conv2d(flow_input)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_conv2d.weight.grad.detach().numpy(), flow_conv2d.weight.grad.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

    def test_maxpool2d(self):
        torch_maxpool2d = torch.nn.MaxPool2d(2, stride=1)
        torch_input = torch.rand((1, 3, 3, 3), requires_grad=True)
        torch_output = torch_maxpool2d(torch_input)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_maxpool2d = MaxPool2d(2)
        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = flow_maxpool2d(flow_input)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

if __name__ == '__main__':
    unittest.main()