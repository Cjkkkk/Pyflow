import unittest
import numpy as np
import flow
import flow.function as F


class TestRefCount(unittest.TestCase):
    # check using gradient_check tools
    def test_ref_count(self):
        a = flow.Tensor(4.1, require_grad=True)
        b = a + 1
        c = a + 2
        d = a + 3
        assert a.ref_count == 3
    
    def test_tensor_used_in_backward(self):
        # if tensor is used in backward, its ref_count should increase 1
        a = flow.Tensor(4.1, require_grad=True)
        b = flow.Tensor(3.1, require_grad=True)
        d = a * b
        assert a.ref_count == 2 # one for mul operation and one for backward of d

    def test_function_inside_function(self):
        # if a autograd function A is called inside of a autograd function B
        # autograd function A should no increase ref_count
        class trivial_function(flow.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return F.add(tensor, flow.Tensor(0))

            @staticmethod
            def backward(ctx, grad):
                return F.add(grad, flow.Tensor(0))

    
        a = flow.Tensor(4.1, require_grad=True)
        b = trivial_function.apply(a)
        assert a.ref_count == 1

if __name__ == '__main__':
    unittest.main()