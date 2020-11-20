import torch
import time
import numpy as np
def cal_fun_t(n, f, *args, **kwargs):
    if n <= 2:
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t_start)
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()

    t_list = []
    for _ in range(n * 2):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize()
        t_list.append(time.perf_counter() - t_start)
    t_list = np.asarray(t_list)
    return t_list[n:].mean()

def assert_equal(a, b, eps):
    with torch.no_grad():
        max_error = (a - b).abs_().max().item()
        assert max_error < eps, f'a={a}, b={b}, max error={max_error} exceeds eps={eps}'
        print(f'max error={max_error}')

