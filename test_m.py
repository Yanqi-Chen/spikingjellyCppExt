import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper.nn

device = 'cuda:7'

asl = wrapper.nn.AutoSparseLinear(2048, 512, bias=False)

asl.benchmark(128, device, verbose=True)
print(asl)