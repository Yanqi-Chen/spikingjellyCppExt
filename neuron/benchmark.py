import torch
import time
import numpy as np
import sys
import math
sys.path.append('.')
import wrapper
import wrapper.neuron

import spikingjelly.clock_driven.neuron as sj_neuron
import spikingjelly.clock_driven.surrogate as sj_surrogate
def cmp_speed():
    def forward_backward(lif, x, T):
        spikes = 0
        for t in range(T):
            spikes += lif(x)
        spikes.sum().backward()
        x.grad.zero_()
        lif.reset()
    
    lif_c = wrapper.neuron.LIFNode(tau=100.0)
    lif_p = sj_neuron.LIFNode(tau=100.0, surrogate_function=sj_surrogate.ATan(alpha=2))
    print(lif_c, lif_p)
    device = 'cuda:6'
    lif_c.to(device)
    lif_p.to(device)

    x = torch.rand([64, 1024], device=device) * 2
    print(x)
    x.requires_grad_(True)

    t_p = wrapper.cal_fun_t(1024, device, forward_backward, lif_p, x, 16)
    x.grad.zero_()
    t_c = wrapper.cal_fun_t(1024, device, forward_backward, lif_c, x, 16)
    x.grad.zero_()

    print(t_c, t_p, 'CUDA speed up =', t_p / t_c)

def cmp_voltage():
    lif_c = wrapper.neuron.LIFNode(tau=100.0)
    lif_p = sj_neuron.LIFNode(tau=100.0, surrogate_function=sj_surrogate.ATan(alpha=2))
    print(lif_c, lif_p)
    device = 'cuda:1'
    lif_c.to(device)
    lif_p.to(device)
    T = 100
    neuron_num = 1024
    x = torch.rand([T, neuron_num], device=device) * 5
    
    with torch.no_grad():
        for t in range(T):
            lif_c(x[t])
            lif_p(x[t])
            print((lif_c.v - lif_p.v).abs_().max().item())
    
        lif_ctt = wrapper.neuron.LIFNodeTT(tau=100.0)
        lif_ctt(x)
        print((lif_c.v - lif_ctt.v).abs_().max().item())
        print(lif_c.v)
        print(lif_ctt.v)
        exit()
        lif_c.reset()
        lif_p.reset()
    s_c = 0
    s_p = 0
    x_c = x.clone()
    x_c.requires_grad_(True)
    x_p = x.clone()
    x_p.requires_grad_(True)
    for t in range(T):
        s_c += lif_c(x_c)
        s_p += lif_p(x_p)
    print(s_c)
    print(s_p)
    s_p.sum().backward()
    print('Python grad', x_p.grad)
    s_c.sum().backward()
    print('CUDA grad', x_c.grad)

cmp_voltage()





