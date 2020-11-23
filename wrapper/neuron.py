import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
use_fast_math = True
extra_cuda_cflags = []
if use_fast_math:
    extra_cuda_cflags.append('-use_fast_math')

cext_neuron = cpp_extension.load(name='neuron', sources=['./neuron/neuron.cpp', './neuron/neuron.cu'], 
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)

class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, detach_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset

        if self.v_reset is None:
            self.v = 0.0
        else:
            self.v = self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def reset(self):
        if self.v_reset is None:
            self.v = 0.0
        else:
            self.v = self.v_reset

class LIFNode(BaseNode):
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau}'
    
    def forward(self, x: torch.Tensor):
        if self.v_reset is None:
            # soft reset
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(x.data)
            spike, self.v = cext_neuron.LIF_soft_reset_forward(x, self.v, self.v_threshold, self.tau)
            return spike
        else:
            # hard reset
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(x.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            spike, self.v = cext_neuron.LIF_hard_reset_forward(x, self.v, self.v_threshold, self.v_reset, self.tau)
            return spike

