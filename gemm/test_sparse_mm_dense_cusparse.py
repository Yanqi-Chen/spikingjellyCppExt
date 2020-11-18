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

cext = cpp_extension.load(name='sparse_mm_dense_cusparse',
                          sources=['./gemm.cpp', './gemm.cu'], verbose=True)





device = 'cuda:0'
batch_size_list = list(range(16, 256 + 1, 16))
batch_size_list.reverse()
in_features_list = list(range(128, 2 ** 15 + 1, 128))
in_features_list.reverse()
out_features_list = list(range(128, 2 ** 15 + 1, 128))
out_features_list.reverse()

sparsity_list = np.arange(0.93, 1, 0.005).tolist()

t_sparse = np.zeros(
    shape=[batch_size_list.__len__(), in_features_list.__len__(), out_features_list.__len__(), sparsity_list.__len__()])
t_dense = np.zeros(shape=[batch_size_list.__len__(), in_features_list.__len__(), out_features_list.__len__()])

idx = 0
for i in range(t_sparse.shape[0]):
    for j in range(t_sparse.shape[1]):
        for k in range(t_sparse.shape[2]):
            batch_size = batch_size_list[i]
            in_features = in_features_list[j]
            out_features = out_features_list[k]
            # print('\n', i, j, k, batch_size, in_features, out_features, end=' ')
            x = torch.rand([batch_size, in_features], device=device)
            w = torch.rand([in_features, out_features], device=device)
            y = torch.zeros([batch_size, out_features], device=device)
            t2 = cal_fun_t(100, torch.mm, x, w)
            t_dense[i][j][k] = t2
            # print(t2, end=' | ')
            for l in range(t_sparse.shape[3]):
                sparsity = sparsity_list[l]
                spike = (x > sparsity).float()
                t1 = cal_fun_t(100, cext.sparse_mm_dense_cusparse, spike, w, y)
                max_error = (spike.mm(w) - y).abs_().max().item()
                # assert max_error < 1e-5, print(max_error)  # 错误检查
                print(batch_size, in_features, out_features, sparsity, max_error)
                t_sparse[i][j][k][l] = t1
                # print(t1, end=' ')

np.save('./sparse_mm_dense_cusparse.npy', {
    't_sparse': t_sparse,
    't_dense': t_dense
})