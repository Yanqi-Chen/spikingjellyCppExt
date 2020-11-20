import torch
import time
import numpy as np
import sys
import math
sys.path.append('..')
import wrapper
import wrapper.functional

def check_sparse_mm_dense(device):
    for i in range(16):
        sparse = (torch.rand([1024, 2048]).to(device) > 0.9).float()
        dense = torch.rand([2048, 4096]).to(device)
        with torch.no_grad():
            wrapper.assert_equal(wrapper.functional.sparse_mm_dense(sparse, dense), sparse.mm(dense), 1e-4)
        sparse.requires_grad_(True)
        dense.requires_grad_(True)
        wrapper.functional.sparse_mm_dense(sparse, dense).sum().backward()
        sparse_grad1 = sparse.grad.clone()
        dense_grad1 = dense.grad.clone()
        sparse.grad.zero_()
        dense.grad.zero_()
        torch.mm(sparse, dense).sum().backward()
        sparse_grad2 = sparse.grad.clone()
        dense_grad2 = dense.grad.clone()
        sparse.grad.zero_()
        dense.grad.zero_()
        wrapper.assert_equal(sparse_grad1, sparse_grad2, 1e-4)
        wrapper.assert_equal(dense_grad1, dense_grad2, 1e-4)



check_sparse_mm_dense('cuda:7')