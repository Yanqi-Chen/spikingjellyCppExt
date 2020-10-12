#include <iostream>
#include <torch/extension.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor spike_matmul_cuda_forward(const torch::Tensor & A, const torch::Tensor & B, torch::Tensor & C);

torch::Tensor spike_matmul_forward(const torch::Tensor & spike, const torch::Tensor & x);

torch::Tensor spike_matmul_forward(const torch::Tensor & spike, const torch::Tensor & x)
{
    CHECK_CUDA(spike);
    CHECK_CUDA(x);
    // spike: [M, N] x:[N, P]
    // y: [M, P]
    auto y = torch::zeros({spike.size(0), x.size(1)}).to(x);
    return spike_matmul_cuda_forward(spike, x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spike_matmul_forward", &spike_matmul_forward);
}

/*
test python code:

from torch.utils import cpp_extension
import torch
import time

cext = cpp_extension.load(name="spike_matmul_forward", 
    sources=["spike_matmul_cuda.cpp", "spike_matmul_cuda_kernel.cu"], verbose=True)

def cal_fun_t(n, f, *args, **kwargs):
    t_start = time.time()
    for _ in range(n):
        f(*args, **kwargs)
    return (time.time() - t_start) / n
device = 'cuda:0'
spike = (torch.rand([1024, 2048]).to(device) > 0.5).float()
x = torch.rand([2048, 4096]).to(device)
t1 = cal_fun_t(100, cext.spike_matmul_forward, spike, x)
t2 = cal_fun_t(100, torch.mm, spike, x)
print(t1, t2, t2 / t1)
*/