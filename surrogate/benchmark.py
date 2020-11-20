import torch
import time
import numpy as np
import sys
import math
sys.path.append('..')
import wrapper
import wrapper.surrogate


def forward_backward(fun, x, alpha):
    fun(x, alpha).sum().backward()

def cmp(fun1, fun2, device, x_shape=[1024], cal_times=1024):
    x = torch.rand(x_shape, device=device)
    alpha = torch.ones([1], device=device)
    with torch.no_grad():
        t1 = wrapper.cal_fun_t(cal_times, fun1, x, alpha)
        t2 = wrapper.cal_fun_t(cal_times, fun2, x, alpha)
    print('forward', t1, t2)
    wrapper.assert_equal(fun1(x, alpha), fun2(x, alpha), 1e-5)
    x.requires_grad_(True)
    t1 = wrapper.cal_fun_t(cal_times, forward_backward, fun1, x, alpha)
    x.grad.zero_()
    t2 = wrapper.cal_fun_t(cal_times, forward_backward, fun2, x, alpha)
    x.grad.zero_()
    print('forward and backward', t1, t2, 1e-5)
    forward_backward(fun1, x, alpha)
    x1_grad = x.grad.clone()
    x.grad.zero_()
    forward_backward(fun2, x, alpha)
    x2_grad = x.grad.clone()
    x.grad.zero_()
    wrapper.assert_equal(x1_grad, x2_grad, 1e-5)



def cmp_sigmoid(device, x_shape=[1024], cal_times=1024):
    class sigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            if x.requires_grad:
                ctx.save_for_backward(x, alpha)
            return x.ge(0).to(x)

        @staticmethod
        def backward(ctx, grad_output):
            grad_x = None
            if ctx.needs_input_grad[0]:
                sgax = (ctx.saved_tensors[0] * ctx.saved_tensors[1]).sigmoid_()
                grad_x = grad_output * (1 - sgax) * sgax * ctx.saved_tensors[1]

            return grad_x, None
    cmp(sigmoid.apply, wrapper.surrogate.sigmoid.apply, device, x_shape, cal_times)
    

def cmp_atan(device, x_shape=[1024], cal_times=1024):
    class atan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            if x.requires_grad:
                ctx.save_for_backward(x, alpha)
            return x.ge(0).to(x)

        @staticmethod
        def backward(ctx, grad_output):
            grad_x = None
            if ctx.needs_input_grad[0]:
                grad_x = ctx.saved_tensors[1] / 2 / (1 + (math.pi / 2 * ctx.saved_tensors[1] * ctx.saved_tensors[0]).pow_(2)) * grad_output

            return grad_x, None
    cmp(atan.apply, wrapper.surrogate.atan.apply, device, x_shape, cal_times)

device = 'cuda:5'
# cmp_sigmoid(device)
# cmp_atan(device, x_shape=[1024], cal_times=1)
x = torch.rand([256], device=device)
alpha = torch.ones([1], device=device)

x.requires_grad_(True)
forward_backward(wrapper.surrogate.atan.apply, x, alpha)
print(x.grad)
x.grad.zero_()

    

