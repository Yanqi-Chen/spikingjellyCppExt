#include <iostream>
#include <torch/extension.h>
#include <math.h>

void mul_cuda(const bool* spike, const float* x, float *y, const int & size, const int & gpu_id);

torch::Tensor mul(torch::Tensor & spike, torch::Tensor & x)
{
    TORCH_CHECK(spike.device().is_cuda(), "spike must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    if (! spike.is_contiguous())
    {
        spike = spike.contiguous();
    }
    if (! x.is_contiguous())
    {
        x = x.contiguous();
    }
    auto y = torch::zeros_like(x.data());
    mul_cuda(spike.data_ptr<bool>(), x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), x.get_device());
    return y;   
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mul", &mul);
}