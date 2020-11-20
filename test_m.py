import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper.nn

device = 'cuda:7'
sparse = (torch.rand([2, 3]).to(device) > 0.9).float()

fc = wrapper.nn.SparseLinear(3, 4, bias=False)
fc.to(device)
y = fc(sparse)
print(y)
y.sum().backward()
print(fc.weight.grad)

print(sparse.grad)

w = fc.weight.data.clone()
w.requires_grad_(True)
y = sparse.mm(w)
print(y)
y.sum().backward()
print(w.grad)