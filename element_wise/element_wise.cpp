#include <iostream>
#include <torch/extension.h>
#include <math.h>

torch::Tensor spikes_or(torch::Tensor & spike, torch::Tensor & x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spikes_or", &spikes_or);
}