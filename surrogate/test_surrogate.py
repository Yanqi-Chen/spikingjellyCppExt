import torch
import time
import numpy as np
import test_m
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

def forward_backward(fun, x, alpha):
    fun(x, alpha).sum().backward()

device = 'cuda:5'
x = torch.rand([1024], device=device)
alpha = torch.ones([1], device=device)
with torch.no_grad():
    t1 = cal_fun_t(1024, sigmoid.apply, x, alpha)
    t2 = cal_fun_t(1024, test_m.sigmoid.apply, x, alpha)

print('forward', t1, t2)

x.requires_grad_(True)
t1 = cal_fun_t(1024, forward_backward, sigmoid.apply, x, alpha)
x.grad.zero_()
t2 = cal_fun_t(1024, forward_backward, test_m.sigmoid.apply, x, alpha)
x.grad.zero_()
print('forward and backward', t1, t2)