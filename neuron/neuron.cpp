#include <iostream>
#include <torch/extension.h>
#include <math.h>

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* v_next, float* s, 
    const float & v_th, const float & v_reset, const float & tau, const int & size);

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
    LIF_hard_reset_forward_cuda(x.data_ptr<float>(), v.data_ptr<float>(), v_next.data_ptr<float>(), s.data_ptr<float>(), 
        v_th, v_reset, tau, x.numel());
    return {s, v_next};
}

void LIF_soft_reset_forward_cuda(const float* x, const float* v, float* v_next, float* s, 
    const float & v_th, const float & tau, const int & size);

std::vector<at::Tensor> LIF_soft_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & tau)
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
    LIF_soft_reset_forward_cuda(x.data_ptr<float>(), v.data_ptr<float>(), v_next.data_ptr<float>(), s.data_ptr<float>(), 
        v_th, tau, x.numel());
    return {s, v_next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
    m.def("LIF_soft_reset_forward", &LIF_soft_reset_forward);
}