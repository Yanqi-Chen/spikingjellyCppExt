import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
import wrapper
from wrapper import neuron
device = 'cuda:1'

if_neuron = neuron.IFNode(v_reset=None)
if_neuron.to(device)
print(if_neuron)
x = torch.rand([4], device=device)
T = 150
print(x)
for t in range(3):
    if_neuron(x)
    print(if_neuron.v)



