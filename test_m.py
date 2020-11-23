import torch
import torch.nn as nn
import torch.nn.functional as F
from wrapper import neuron
device = 'cuda:2'


lif = neuron.LIFNode()
lif.to(device)
print(lif)
x = torch.rand([4], device=device) * 10
T = 150
for t in range(1):
    lif(x)
    print(lif.v)



