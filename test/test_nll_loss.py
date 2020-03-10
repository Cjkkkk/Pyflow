import torch
import torch.nn as nn
import flow.function as F
from flow.tensor import Tensor

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
input = torch.tensor([[0, 1, 2], [1, 2, 3]], requires_grad=True, dtype=torch.float)
r = m(input)
r.retain_grad()
target = torch.tensor([1, 2])
output = loss(r, target)
output.backward()
print(output)
print(r.grad)

ll = Tensor(r.detach().numpy(), require_grad=True)
t = Tensor(target.detach().numpy())
result = F.nll_loss(ll, t, True)
result.backward()
print(result)
print(ll.grad)