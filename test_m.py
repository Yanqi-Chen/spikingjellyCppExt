from spikingjelly import cext
from spikingjelly.cext import neuron as cext_neuron
from spikingjelly.clock_driven import neuron, surrogate, layer
import torch

def cal_forward_t(multi_step_neuron, x, repeat_times):
    with torch.no_grad():
        used_t = cext.cal_fun_t(repeat_times, x.device, multi_step_neuron, x)
        multi_step_neuron.reset()
        return used_t

def forward_backward(multi_step_neuron, x):
    multi_step_neuron(x).sum().backward()
    multi_step_neuron.reset()
    x.grad.zero_()

def cal_forward_backward_t(multi_step_neuron, x, repeat_times):
    x.requires_grad_(True)
    used_t = cext.cal_fun_t(repeat_times, x.device, forward_backward, multi_step_neuron, x)
    return used_t

device = 'cuda:0'
lif = layer.MultiStepContainer(neuron.LIFNode(surrogate_function=surrogate.ATan(alpha=2.0)))
lif_cuda = layer.MultiStepContainer(cext_neuron.LIFNode(surrogate_function='ATan', alpha=2.0))
lif_cuda_tt = cext_neuron.MultiStepLIFNode(surrogate_function='ATan', alpha=2.0)
lif.to(device)
lif_cuda.to(device)
lif_cuda_tt.to(device)
N = 128 * 16 * 16
T = 64
x = torch.rand(T, N, device=device)
print(cal_forward_t(lif, x, 1024))
print(cal_forward_t(lif_cuda, x, 1024))
print(cal_forward_t(lif_cuda_tt, x, 1024))

print(cal_forward_backward_t(lif, x, 1024))
print(cal_forward_backward_t(lif_cuda, x, 1024))
print(cal_forward_backward_t(lif_cuda_tt, x, 1024))