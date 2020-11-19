import torch
import time

def cal_fun_t(n, f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()

    t_list = []
    for _ in range(n):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize()
        t_list.append(time.perf_counter() - t_start)
    t_list = np.asarray(t_list)
    return (t_list.sum() - t_list.max() - t_list.min()) / (n - 2)

