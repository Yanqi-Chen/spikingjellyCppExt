import torch
import torch.nn as nn
import torch.nn.functional as F
import wrapper.layer

device = 'cuda:7'

asl = wrapper.layer.AutoSparseLinear(2048, 512, bias=False)
asl.to(device)

asl(torch.rand([16, 2048], device=device))
asl(torch.rand([32, 2048], device=device))
asl(torch.rand([16, 2048], device=device))


