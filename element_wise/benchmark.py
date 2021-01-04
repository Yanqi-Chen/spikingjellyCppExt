import torch
import time
import numpy as np
import sys
sys.path.append('.')
import math
from torch.utils import cpp_extension
import wrapper

cext_element_wise = cpp_extension.load(name='cext_element_wise',
    sources=['./element_wise/element_wise.cpp', './element_wise/element_wise.cu'], verbose=True)

def test_spikes_or():
    def torch_or(x, y):
        return torch.clamp(x + y, 0, 1)
    device = 'cuda:0'
    N = 2**20
    
    x = (torch.rand([N], device=device) > 0.5).float()
    y = (torch.rand([N], device=device) > 0.5).float()
    with torch.no_grad():
        t1 = wrapper.cal_fun_t(1024, device, torch_or, x, y)
        t2 = wrapper.cal_fun_t(1024, device, cext_element_wise.spikes_or, x, y)
        print(t1, t2, t1 / t2)
        wrapper.assert_equal(torch_or(x, y), cext_element_wise.spikes_or(x, y), 1e-9)
    

test_spikes_or()
