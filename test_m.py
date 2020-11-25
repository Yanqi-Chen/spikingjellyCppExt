import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
import wrapper
device = 'cuda:1'

cext_element_wise_mul = cpp_extension.load(name='element_wise',
    sources=['./element_wise/element_wise.cpp', './element_wise/element_wise.cu'], verbose=True).mul

x = torch.rand([16, 2048], device=device)
spike = (x > 0.5).float()
t1 = wrapper.cal_fun_t(1024, cext_element_wise_mul, spike.bool(), x)
t2 = wrapper.cal_fun_t(1024, torch.mul, spike, x)
t3 = wrapper.cal_fun_t(1024, torch.mul, spike.bool(), x)
print(t1, t2, t3)
exit()

from wrapper import neuron
device = 'cuda:0'


lif = neuron.LIFNode(tau=100.0)
lif.to(device)
print(lif)
x = torch.rand([4], device=device) * 10
T = 150
for t in range(T):
    lif(x)
    print(lif.v)



