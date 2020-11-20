import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper.layer

device = 'cuda:7'

asl = wrapper.layer.AutoSparseLinear(2048, 512, bias=False)

asl.benchmark(128, device, run_times=1024, verbose=True, precision=1e-6)
print(asl)