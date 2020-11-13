from torch.utils import cpp_extension
import torch
import time

cext = cpp_extension.load(name='sparse_mm_dense_cusparse', 
    sources=['./gemm.cpp', './gemm.cu'], verbose=True)

def cal_fun_t(n, f, *args, **kwargs):
    torch.cuda.synchronize()
    t_start = time.time()
    # warm up
    f(*args, **kwargs)
    for _ in range(n):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - t_start) / n

device = 'cpu'
# torch.cuda.set_device(device)
mat_size = 2**11
sparsity = 0.99
spike = (torch.rand([mat_size, mat_size]).to(device) > sparsity).float()
x = torch.rand([mat_size, mat_size]).to(device)
y = torch.zeros([mat_size, mat_size]).to(device)
t1 = cal_fun_t(100, cext.sparse_mm_dense_cusparse, spike, x, y)
print('manual:\n', y)
t2 = cal_fun_t(100, torch.mm, spike, x)
print('pytorch:\n', spike.mm(x))

print(t1, t2, 'manual_speed / pytorch_speed = ', t2 / t1)



