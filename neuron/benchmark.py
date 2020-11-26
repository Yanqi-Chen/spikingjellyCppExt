import torch
import time
import numpy as np
import sys
import math
sys.path.append('.')
import wrapper
import wrapper.neuron

import spikingjelly.clock_driven.neuron as sj_neuron

def cmp_speed():
    lif_c = wrapper.neuron.LIFNode()
    lif_p = sj_neuron.LIFNode()
    print(lif_c, lif_p)
    device = 'cuda:1'
    lif_c.to(device)
    lif_p.to(device)
    x = torch.rand([64, 1024], device=device) * 2

    with torch.no_grad():
        t_c = wrapper.cal_fun_t(1024, lif_c, x)
        t_p = wrapper.cal_fun_t(1024, lif_p, x)
        print(t_c, t_p, 'CUDA speed up =', t_p / t_c)

def cmp_voltage():
    lif_c = wrapper.neuron.LIFNode(tau=100.0)
    lif_p = sj_neuron.LIFNode(tau=100.0)
    print(lif_c, lif_p)
    device = 'cuda:1'
    lif_c.to(device)
    lif_p.to(device)
    x = torch.rand([1024], device=device)
    T = 100
    for t in range(T):
        lif_c(x)
        lif_p(x)
        print((lif_c.v - lif_p.v).abs_().max().item())

cmp_speed()