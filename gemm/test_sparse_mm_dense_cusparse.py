from torch.utils import cpp_extension
import torch
import time
import numpy as np

cext = cpp_extension.load(name='sparse_mm_dense_cusparse', 
    sources=['./gemm.cpp', './gemm.cu'], verbose=True)

def cal_fun_t(n, f, *args, **kwargs):
    
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(n):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - t_start) / n

device = 'cuda:0'
mat_size_list = list(range(128, 16384 + 1, 128))
mat_size_list.reverse()
sparsity_list = np.arange(0.93, 1, 0.005).tolist()
# sparsity_list.reverse()
results = np.zeros(shape=[mat_size_list.__len__() * sparsity_list.__len__(), 4])
idx = 0
for mat_size in mat_size_list:
    a = torch.rand([mat_size, mat_size]).to(device)
    x = torch.rand([mat_size, mat_size]).to(device)
    y = torch.zeros([mat_size, mat_size]).to(device)
    for sparsity in sparsity_list:
        spike = (a > sparsity).float()
        t1 = cal_fun_t(100, cext.sparse_mm_dense_cusparse, spike, x, y)
        t2 = cal_fun_t(100, torch.mm, spike, x)
        results[idx][0] = mat_size
        results[idx][1] = sparsity
        results[idx][2] = t1
        results[idx][3] = t2
        print(idx, '/', results.shape[0], mat_size, '%.4f'%sparsity, t1, t2)
        idx += 1
np.savetxt('./sparse_mm_dense_cusparse.csv', results, delimiter=',')