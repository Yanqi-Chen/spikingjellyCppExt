from torch.utils import cpp_extension
import torch
import time

cext = cpp_extension.load(name='sparse_mm_dense_cusparse', 
    sources=['./gemm.cpp', './gemm.cu'], verbose=True)

def cal_fun_t(n, f, *args, **kwargs):
    t_start = time.time()
    # warm up
    f(*args, **kwargs)
    for _ in range(n):
        f(*args, **kwargs)
    return (time.time() - t_start) / n

device = 'cuda:0'
mat_size = 2**12
spike = (torch.rand([mat_size, mat_size]).to(device) > 0.99).float()
x = torch.rand([mat_size, mat_size]).to(device)
y = torch.zeros([mat_size, mat_size]).to(device)
t1 = cal_fun_t(100, cext.sparse_mm_dense_cusparse, spike, x, y)
print('manual:\n', y)
t2 = cal_fun_t(100, torch.mm, spike, x)
print('pytorch:\n', spike.mm(x))

print(t1, t2, t2 / t1)



