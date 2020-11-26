import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
import wrapper
from wrapper import neuron
device = 'cuda:1'

lif = neuron.LIFNode(tau=100.0)
lif.to(device)
print(lif)
x = torch.rand([4], device=device)
T = 150
print(x)
for t in range(1):
    lif(x)
    print(lif.v)



