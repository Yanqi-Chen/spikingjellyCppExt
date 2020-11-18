from torch.utils import cpp_extension
import torch
import time
import numpy as np

def cal_fun_t(n, f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(n):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - t_start) / n

cext_surrogate = cpp_extension.load(name='surrogate',
                          sources=['./surrogate.cpp'], verbose=True)

device = 'cuda:0'
x = torch.rand([8], device=device)
alpha = torch.ones([1], device=device)
x.requires_grad_(True)
y = cext_surrogate.atan(x, alpha)
print(y)
y.sum().backward()
print(x.grad)
