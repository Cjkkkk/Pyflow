import unittest
import numpy as np
import torch
import torch.nn.functional as PF
import torch.optim as torch_optim

import flow.function as F
from flow.tensor import Tensor, randn
from flow.utils import gradient_check
from flow.module import Conv2d
import flow.optim as flow_optim


class TestGradientAuto(unittest.TestCase):
    # check using gradient_check tools
    def test_max_auto(self):
        gradient_check(F.max, randn((5,4,3), require_grad=True), axis=1)
        gradient_check(F.max, randn((5,4), require_grad=True), axis=None)

    def test_min_auto(self):
        gradient_check(F.min, randn((5,4), require_grad=True), axis=1)
        gradient_check(F.min, randn((5,4), require_grad=True), axis=None)

    def test_sum_auto(self):
        gradient_check(F.sum, randn((5,4), require_grad=True))

    def test_nll_loss_auto(self):
        gradient_check(F.nll_loss, randn((2,4), require_grad=True), Tensor([0, 1]))

    def test_square_loss_auto(self):
        gradient_check(F.square_loss, randn((2,4), require_grad=True), randn((2,4), require_grad=True))

    def test_logsoftmax_auto(self):
        gradient_check(F.log_softmax, randn((5,4), require_grad=True), dim=1)
    
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
        flow_output = F.log_softmax(flow_input, dim=1)
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

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.max_pool2d(flow_input, 2, 1)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)
    
    def test_nll_loss(self):
        torch_input = torch.rand((4, 4), requires_grad=True)
        torch_output = PF.nll_loss(torch_input, torch.tensor([0, 1, 2, 3]))
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.nll_loss(flow_input, Tensor([0, 1, 2, 3]))
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

        torch_input = torch.rand((4, 4), requires_grad=True)
        torch_output = PF.nll_loss(torch_input, torch.tensor([0, 1, 2, 3]), reduction="sum")
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.nll_loss(flow_input, Tensor([0, 1, 2, 3]), reduction="sum")
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)


    def test_relu(self):
        torch_input = torch.rand((1, 3, 3, 3), requires_grad=True)
        torch_output = PF.relu(torch_input)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.relu(flow_input)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

    def test_min(self):
        torch_input = torch.rand((3, 5), requires_grad=True)
        torch_output = torch.min(torch_input)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.min(flow_input, axis=None)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

        torch_input = torch.rand((3, 5), requires_grad=True)
        torch_output, _ = torch.min(torch_input, dim=1)
        torch_output.backward(torch.ones(torch_output.shape))

        flow_input = Tensor(torch_input.detach().numpy(), require_grad=True)
        flow_output = F.min(flow_input, axis=1)
        flow_output.backward()

        assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)
        assert np.allclose(torch_input.grad.detach().numpy(), flow_input.grad.data, atol=1e-6)

    def test_optim_SGD(self):
        for i in range(10):
            torch_input = torch.rand((2, 2), requires_grad=True)
            flow_input = Tensor(torch_input.detach().numpy().copy(), require_grad=True)

            torch_sgd = torch_optim.SGD([torch_input], lr=0.01)
            torch_output, _ = torch.min(torch_input, dim=1)
            torch_output.backward(torch.ones(torch_output.shape))
            torch_sgd.step()
            torch_sgd.zero_grad()

            flow_sgd = flow_optim.SGD([flow_input], lr=0.01)
            flow_output = F.min(flow_input, axis=1)
            flow_output.backward()
            flow_sgd.step()
            flow_sgd.zero_grad()
            
            assert np.allclose(torch_output.detach().numpy(), flow_output.data, atol=1e-6)

if __name__ == '__main__':
    unittest.main()