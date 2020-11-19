import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import cpp_extension
import math
cext_surrogate = cpp_extension.load(name='surrogate', sources=['../surrogate/surrogate.cpp', '../surrogate/surrogate.cu'], verbose=True)

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return x.ge(0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = ctx.saved_tensors[1] * cext_surrogate.sigmoid_backward(grad_output, (ctx.saved_tensors[0] * ctx.saved_tensors[1]).sigmoid_())
        return grad_x, None

