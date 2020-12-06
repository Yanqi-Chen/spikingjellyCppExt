import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
use_fast_math = True
extra_cuda_cflags = []
if use_fast_math:
    extra_cuda_cflags.append('-use_fast_math')

cext_neuron_forward = cpp_extension.load(name='neuron_forward', sources=['./neuron/neuron_forward.cpp', './neuron/neuron_forward_kernel.cu'], 
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)

cext_neuron_backward = cpp_extension.load(name='neuron_backward', sources=['./neuron/neuron_backward.cpp', './neuron/neuron_backward_kernel.cu'], 
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)

class lif_hard_forward_backward_atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, tau, alpha, detach_reset):
        h, spike, v_next = cext_neuron_forward.LIF_hard_reset_forward(x, v, v_threshold, v_reset, tau)
        if x.requires_grad:
            ctx.save_for_backward(h, spike)
            ctx.v_threshold = v_threshold
            ctx.v_reset = v_reset
            ctx.tau = tau
            ctx.alpha = alpha
            ctx.detach_reset = detach_reset
        return h, spike, v_next

    @staticmethod
    def backward(ctx, grad_h, grad_spike, grad_v_next):
        grad_x, grad_v = cext_neuron_backward.LIF_hard_reset_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.v_threshold, ctx.v_reset, ctx.alpha, ctx.detach_reset, 0, ctx.tau)
        return grad_x, grad_v, None, None, None, None, None


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
            raise NotImplementedError
        else:
            # hard reset
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(x.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            h, spike, self.v = lif_hard_forward_backward_atan.apply(x, self.v, self.v_threshold, self.v_reset, self.tau, 2.0, self.detach_reset)
            return spike

class LIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, tau):
        if v_reset is None:
            raise NotImplementedError
        h_seq, spike_seq, v_next = cext_neuron_forward.LIF_hard_reset_fptt(x_seq, v, v_threshold, v_reset, tau)
        if x_seq.requires_grad:
            ctx.save_for_backward(h_seq, spike_seq)
            ctx.v_threshold = v_threshold
            ctx.v_reset = v_reset
            ctx.alpha = alpha
            ctx.detach_reset = detach_reset
            ctx.grad_surrogate_function_index = grad_surrogate_function_index
            ctx.tau = tau
        return h_seq, spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_h_seq, grad_spike_seq, grad_v_next_seq):
        grad_x, grad_v = cext_neuron_backward.LIF_hard_reset_bptt(grad_spike_seq, grad_v_next_seq, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.v_threshold, ctx.v_reset, ctx.alpha, ctx.detach_reset, ctx.grad_surrogate_function_index, ctx.tau)
        return grad_x, grad_v, None, None, None, None, None, None

class LIFNodeTT(LIFNode):
     def forward(self, x: torch.Tensor):
        if self.v_reset is None:
            # soft reset
            raise NotImplementedError
        else:
            # hard reset
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(x[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            h_seq, spike_seq, self.v = LIFMultiStep.apply(x, self.v, self.v_threshold, self.v_reset, 2.0, self.detach_reset, 0, self.tau)
            return spike_seq