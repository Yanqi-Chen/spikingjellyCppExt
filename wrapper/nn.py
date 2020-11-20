import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper
import wrapper.functional
import math
import numpy as np
import time
class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return wrapper.functional.sparse_mm_dense(sparse, self.weight)
        else:
            return wrapper.functional.sparse_mm_dense(sparse, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class AutoSparseLinear(SparseLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
        self.critical_sparsity = {}  
        # 键是输入数据的batch_size，值是临界稀疏度
        # 当稀疏度高于临界稀疏度，前向传播使用稀疏矩阵乘法；否则使用普通矩阵乘法

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, critical_sparsity={self.critical_sparsity}'

    def benchmark(self, batch_size: int, device=None, run_times=1024, verbose=False):
        if verbose:
            print('AutoSparseLinear is running benchmark...')
        if device is None:
            device = self.weight.device
        
        if self.bias is None:
            bias = False
        else:
            bias = True
        
        fc_sparse = SparseLinear(self.in_features, self.out_features, bias)
        fc_sparse.to(device)
        fc_dense = nn.Linear(self.in_features, self.out_features, bias)
        fc_dense.to(device)

        sparisity_r = 1.0
        sparisity_l = 0.9
        # 二分查找临界稀疏度
        
        while True:
            sparisity = (sparisity_l + sparisity_r) / 2
            x = torch.rand(size=[batch_size, self.in_features], device=device)
            sparse = (x > sparisity).to(x)
            sparisity_a = 1 - (sparse != 0).to(x).mean().item()  # sparse的真实稀疏度

            # 计算稀疏前反向所需时间
            t_list = []
            for _ in range(run_times * 2):
                fc_sparse.zero_grad()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                fc_sparse(x).sum().backward()
                torch.cuda.synchronize()
                t_list.append(time.perf_counter() - t_start)
            t_list = np.asarray(t_list)
            t_sparse = t_list[run_times:].sum()

            # 计算稠密前反向所需时间
            t_list = []
            for _ in range(run_times * 2):
                fc_dense.zero_grad()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                fc_dense(x).sum().backward()
                torch.cuda.synchronize()
                t_list.append(time.perf_counter() - t_start)
            t_list = np.asarray(t_list)
            t_dense = t_list[run_times:].sum()
            if verbose:
                print(f'sparisity_a={sparisity_a}, t_sparse={t_sparse}, t_dense={t_dense}')
            
            if t_sparse > t_dense:
                sparisity_l = sparisity_a
            elif t_sparse < t_dense:
                sparisity_r = sparisity_a
            else:
                break

            if sparisity_r - sparisity_l  < 1e-4:
                break
        
        self.critical_sparsity[str(batch_size)] = sparisity_a



                
                




                


