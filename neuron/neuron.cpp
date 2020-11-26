#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include "neuron_def.h"
DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau);
DEF_HARD_RESET_FORWARD_FUNCTION(LIF_hard_reset_forward, const float & tau)
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
    auto spike = torch::zeros_like(v.data());
    if (! v_next.is_contiguous())
    {
        v_next = v_next.contiguous();
    }
    if (! spike.is_contiguous())
    {
        spike = spike.contiguous();
    }
    LIF_hard_reset_forward_cuda(x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), 
        v_th, v_reset, x.numel(), x.get_device(), tau);
    
    return {spike, v_next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
}