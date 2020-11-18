import torch
import time

def cal_fun_t(n, f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(n):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - t_start) / n

