import unittest
import numpy as np
import flow

class TestInPlace(unittest.TestCase):
    # check using gradient_check tools
    def test_iadd(self):
        a = flow.Tensor(3)
        b = a
        a += flow.Tensor(5)
        assert np.allclose(a.data, b.data)

        a = flow.Tensor(3)
        b = a
        a = a + flow.Tensor(5)
        assert not np.allclose(a.data, b.data)

    def test_isub(self):
        a = flow.Tensor(3)
        b = a
        a -= flow.Tensor(5)
        assert np.allclose(a.data, b.data)

        a = flow.Tensor(3)
        b = a
        a = a - flow.Tensor(5)
        assert not np.allclose(a.data, b.data)
    
    def test_imul(self):
        a = flow.Tensor(3)
        b = a
        a *= flow.Tensor(5)
        assert np.allclose(a.data, b.data)

        a = flow.Tensor(3)
        b = a
        a = a * flow.Tensor(5)
        assert not np.allclose(a.data, b.data)
    
    def test_itrue_div(self):
        # inplace div will cause a dtype change from int to float, which raise error
        # therefore define a to be 3.1 in the first place
        a = flow.Tensor(3.1)
        b = a
        a /= flow.Tensor(5)
        assert np.allclose(a.data, b.data)

        a = flow.Tensor(3)
        b = a
        a = a / flow.Tensor(5)
        assert not np.allclose(a.data, b.data)

    # def test_inplace_gradient(self):
    #     a = flow.Tensor(3.1, require_grad=True)
    #     b = a + flow.Tensor(1.1)
    #     b += flow.Tensor(5)
    #     b *= flow.Tensor(5)
    #     b.backward()
    #     assert np.allclose(b.grad.data, np.array(5))

if __name__ == '__main__':
    unittest.main()