#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include "neuron_def.h"

void LIF_hard_reset_backward_cuda(
  float* grad_x, float* grad_v,
  const float* grad_spike, const float* grad_v_next,
  const float* x, const float* h, const float* spike, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index, 
  const float & tau);

#define HARD_RESET_BACKWARD CHECK_TENSOR(grad_spike);CHECK_TENSOR(grad_v_next);CHECK_TENSOR(x);CHECK_TENSOR(h);CHECK_TENSOR(spike);auto grad_x = torch::zeros_like(x.data());auto grad_v = torch::zeros_like(x.data());CHECK_TENSOR(grad_x);CHECK_TENSOR(grad_v)

std::vector<at::Tensor> LIF_hard_reset_backward(
    torch::Tensor & grad_spike, torch::Tensor & grad_v_next,
    torch::Tensor & x, torch::Tensor & h, torch::Tensor & spike,
    const float & v_th, const float & v_reset, 
    const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index,
    const float & tau)
{
    HARD_RESET_BACKWARD;
    LIF_hard_reset_backward_cuda(
        grad_x.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_spike.data_ptr<float>(), grad_v_next.data_ptr<float>(),
        x.data_ptr<float>(), h.data_ptr<float>(), spike.data_ptr<float>(),
        v_th, v_reset, x.numel(), x.get_device(),
        alpha, detach_reset, grad_surrogate_function_index,
        tau);
    return {grad_x, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_backward", &LIF_hard_reset_backward);
}

