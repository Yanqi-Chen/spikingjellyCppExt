import torch
import torch.nn as nn
import torch.nn.functional as F
from wrapper import neuron
device = 'cuda:2'


lif = neuron.LIFNode(tau=100.0)
lif.to(device)
print(lif)
x = torch.rand([4], device=device) * 10
T = 150
for t in range(T):
    lif(x)
    print(lif.v)



