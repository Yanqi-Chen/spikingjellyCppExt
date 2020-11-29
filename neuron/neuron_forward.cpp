#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include "neuron_def.h"

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* h, float* spike, float* v_next, 
    const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
    const float & tau);

std::vector<at::Tensor> LIF_hard_reset_forward(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset, 
    const float & tau)
{
    CHECK_TENSOR(x);
    CHECK_TENSOR(v);
    auto h = torch::zeros_like(v.data());
    auto spike = torch::zeros_like(v.data());
    auto v_next = torch::zeros_like(v.data());
    CHECK_TENSOR(h);
    CHECK_TENSOR(spike);
    CHECK_TENSOR(v_next);
    
    LIF_hard_reset_forward_cuda(x.data_ptr<float>(), v.data_ptr<float>(), h.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), 
        v_th, v_reset, x.numel(), x.get_device(),
        tau);
    return {h, spike, v_next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
}

