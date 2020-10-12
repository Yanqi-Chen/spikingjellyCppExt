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



