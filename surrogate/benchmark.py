import torch
import time
import numpy as np
import sys
sys.path.append('..')
import wrapper.surrogate

def cal_fun_t(n, f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)
    torch.cuda.synchronize()

    t_list = []
    for _ in range(n):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        f(*args, **kwargs)
        torch.cuda.synchronize()
        t_list.append(time.perf_counter() - t_start)
    t_list = np.asarray(t_list)
    return (t_list.sum() - t_list.max() - t_list.min()) / (n - 2)

def forward_backward(fun, x, alpha):
    fun(x, alpha).sum().backward()

def cmp(fun1, fun2, device, x_shape=[1024], cal_times=1024):
    x = torch.rand(x_shape, device=device)
    alpha = torch.ones([1], device=device)
    with torch.no_grad():
        t1 = cal_fun_t(cal_times, fun1, x, alpha)
        t2 = cal_fun_t(cal_times, fun2, x, alpha)
    print('forward', t1, t2)
    x.requires_grad_(True)
    t1 = cal_fun_t(cal_times, forward_backward, fun1, x, alpha)
    x.grad.zero_()
    t2 = cal_fun_t(cal_times, forward_backward, fun2, x, alpha)
    x.grad.zero_()
    print('forward and backward', t1, t2)


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
    

def cmp_atan():
    class atan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            if x.requires_grad:
                ctx.save_for_backward(x, alpha)
            return heaviside(x)

        @staticmethod
        def backward(ctx, grad_output):
            grad_x = None
            if ctx.needs_input_grad[0]:
                grad_x = ctx.saved_tensors[1] / 2 / (1 + (math.pi / 2 * ctx.saved_tensors[1] * ctx.saved_tensors[0]).pow_(2)) * grad_output

            return grad_x, None
    raise NotImplementedError


cmp_sigmoid('cuda:5')


    

