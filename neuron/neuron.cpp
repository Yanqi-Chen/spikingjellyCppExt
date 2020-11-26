#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include "neuron_def.h"
DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau);
DEF_HARD_RESET_FORWARD_FUNCTION(LIF_hard_reset_forward, const float & tau)
{
    CALL_HARD_RESET_FORWARD_CUDA_FUNCTION(LIF_hard_reset_forward_cuda, tau);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LIF_hard_reset_forward", &LIF_hard_reset_forward);
}