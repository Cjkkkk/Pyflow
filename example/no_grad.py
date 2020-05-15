import flow.tensor as tensor
import flow.autograd as autograd


a = tensor.Tensor([1.0, 2.0, 3.0], require_grad=True)
b = a + 1
print(b.require_grad, b.grad_fn, b.is_leaf)

with autograd.no_grad():
    b = a + 1
    print(b.require_grad, b.grad_fn, b.is_leaf)