import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils import cpp_extension
use_fast_math = True
extra_cuda_cflags = []
if use_fast_math:
    extra_cuda_cflags.append('-use_fast_math')

cext_neuron = cpp_extension.load(name='neuron', sources=['./neuron/neuron.cpp', './neuron/neuron_forward_kernel.cu', './neuron/neuron_backward_kernel.cu'], 
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True)



class LIFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau):
        if v_reset is None:
            raise NotImplementedError

        spike, v_next, grad_s_to_h, grad_v_to_h = cext_neuron.LIF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        ctx.reciprocal_tau = reciprocal_tau

        return spike, v_next

    @staticmethod
    def backward(ctx, grad_spike, grad_v_next):
        grad_x, grad_v = cext_neuron.LIF_hard_reset_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.reciprocal_tau)
        return grad_x, grad_v, None, None, None, None, None, None

class LIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau):
        if v_reset is None:
            raise NotImplementedError

        spike_seq, v_next, grad_s_to_h, grad_v_to_h = cext_neuron.LIF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        ctx.reciprocal_tau = reciprocal_tau
        return spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_next):
        grad_x, grad_v = cext_neuron.LIF_hard_reset_bptt(grad_spike_seq, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.reciprocal_tau)
        return grad_x, grad_v, None, None, None, None, None, None

surrogate_function_dict = {
    'ATan': 0,
    'Sigmoid': 1
}
class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.grad_surrogate_function_index = surrogate_function_dict[surrogate_function]
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.reset()
    
    def reset(self):
        self.v = self.v_reset

class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        self.reciprocal_tau = 1 / tau
    
    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike, self.v = LIFStep.apply(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index, self.reciprocal_tau)
            else:
                spike, self.v = cext_neuron.LIF_hard_reset_forward(dv, self.v, self.v_threshold, self.v_reset, self.reciprocal_tau)
            return spike

class MultiStepLIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        self.reciprocal_tau = 1 / tau

    def forward(self, dv_seq: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv_seq[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike_seq, self.v = LIFMultiStep.apply(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha,
                                                       self.detach_reset, self.grad_surrogate_function_index, self.reciprocal_tau)
            else:
                spike_seq, self.v = cext_neuron.LIF_hard_reset_fptt(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index, self.reciprocal_tau)
            return spike_seq

class ParametricLIFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau):
        if v_reset is None:
            raise NotImplementedError

        spike, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau = cext_neuron.ParametricLIF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h, grad_h_to_rtau)
        ctx.reciprocal_tau = reciprocal_tau

        return spike, v_next

    @staticmethod
    def backward(ctx, grad_spike, grad_v_next):
        grad_x, grad_v, grad_rtau = cext_neuron.ParametricLIF_hard_reset_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.reciprocal_tau)
        return grad_x, grad_v, None, None, None, None, None, grad_rtau

class ParametricLIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau):
        if v_reset is None:
            raise NotImplementedError

        spike_seq, v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau = cext_neuron.ParametricLIF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h, grad_h_to_rtau)
        ctx.reciprocal_tau = reciprocal_tau

        return spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_next):
        grad_x_seq, grad_v, grad_rtau = cext_neuron.ParametricLIF_hard_reset_bptt(grad_spike_seq, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.saved_tensors[2], ctx.reciprocal_tau)
        return grad_x_seq, grad_v, None, None, None, None, None, grad_rtau

class ParametricLIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0,
                 detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        self.w = nn.Parameter(torch.Tensor([- math.log(tau - 1)]))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike, self.v = ParametricLIFStep.apply(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset,
                                              self.grad_surrogate_function_index, self.w.sigmoid())
            else:
                spike, self.v = cext_neuron.LIF_hard_reset_forward(dv, self.v, self.v_threshold, self.v_reset,
                                                                   self.w.sigmoid())
            return spike

class MultiStepParametricLIFNode(ParametricLIFNode):

    def forward(self, dv_seq: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv_seq[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike_seq, self.v = ParametricLIFMultiStep.apply(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha,
                                                       self.detach_reset, self.grad_surrogate_function_index, self.w.sigmoid())
            else:
                spike_seq, self.v = cext_neuron.LIF_hard_reset_fptt(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index, self.w.sigmoid())
            return spike_seq

from spikingjelly.clock_driven import neuron, surrogate
class PLIFNode(neuron.BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=False,
                 monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.w = nn.Parameter(torch.Tensor([- math.log(tau - 1)]))

    def neuronal_charge(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()