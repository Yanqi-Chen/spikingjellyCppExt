#include <iostream>
#include <torch/extension.h>
#include <math.h>
torch::Tensor alpha_sigmoid_backward(const torch::Tensor & grad_output, const torch::Tensor & x, const torch::Tensor & alpha)
{
    return alpha * torch::sigmoid_backward(grad_output, torch::sigmoid(x * alpha));
}

void alpha_atan_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size);

torch::Tensor alpha_atan_backward(const torch::Tensor & grad_output, const torch::Tensor & x, const torch::Tensor & alpha)
{   
    if (x.get_device() < 0)
    {
        // CPU
        return alpha / 2.0f / (1.0f + (M_PI_2 * alpha * x).pow_(2)) * grad_output;
    }
    else
    {   
        // gpu
        TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
        auto grad_x = torch::zeros_like(x.data());
        alpha_atan_backward_cuda(grad_output.data_ptr<float>(), x.data_ptr<float>(), alpha.item<float>(), grad_x.data_ptr<float>(), x.numel());
        return grad_x;
    }

}
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_backward", &torch::sigmoid_backward);
    m.def("alpha_sigmoid_backward", &alpha_sigmoid_backward);
    m.def("alpha_atan_backward", &alpha_atan_backward);
}