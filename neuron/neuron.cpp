#include <iostream>
#include <torch/extension.h>
#include <math.h>
// 定义前向传播函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_FORWARD_FUNCTION(function_name, ...) void function_name (const float* x, const float* v, float* s, float* v_next, const float & v_th, const float & v_reset, const int & size, const int & gpu_id, __VA_ARGS__)
DEF_FORWARD_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau);


std::vector<at::Tensor> LIF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset, const float & tau)
{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.device().is_cuda(), "v must be a CUDA tensor");
    if (! x.is_contiguous())
    {
        x = x.contiguous();
    }
    if (! v.is_contiguous())
    {
        v = v.contiguous();
    }
    auto v_next = torch::zeros_like(v.data());
    auto s = torch::zeros_like(v.data());
    if (! v_next.is_contiguous())
    {
        v_next = v_next.contiguous();
    }
    if (! s.is_contiguous())
    {
        s = s.contiguous();
    }
    LIF_hard_reset_forward_cuda(x.data_ptr<float>(), v.data_ptr<float>(), s.data_ptr<float>(), v_next.data_ptr<float>(), 
        v_th, v_reset, x.numel(), x.get_device(), tau);
    return {s, v_next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
}