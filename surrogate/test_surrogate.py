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

cext_sigmoid = cpp_extension.load(name='sigmoid',
                          sources=['./surrogate.cpp'], verbose=True)

device = 'cuda:0'