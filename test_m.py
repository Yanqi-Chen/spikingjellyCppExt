import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper.nn

device = 'cuda:7'

asl = wrapper.nn.AutoSparseLinear(2048, 4096, bias=False)

asl.benchmark(2048, device, verbose=True)
print(asl)