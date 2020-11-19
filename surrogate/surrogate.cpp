#include <iostream>
#include <torch/extension.h>
#include <math.h>
using namespace torch::autograd;
torch::Tensor alpha_sigmoid_backward(const torch::Tensor & grad_output, const torch::Tensor & x, const torch::Tensor & alpha)
{
    return alpha * torch::sigmoid_backward(grad_output, torch::sigmoid(x * alpha));
}

void atan_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size);

torch::Tensor atan_backward(const torch::Tensor & grad_output, const torch::Tensor & x, const torch::Tensor & alpha)
{   
    if (x.get_device() < 0)
    {
        // CPU
        return alpha / 2 / (1 + M_PI_2 * alpha * x).pow_(2) * grad_output;
    }
    else
    {   
        // gpu
        TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
        auto grad_x = torch::zeros_like(x);
        atan_backward_cuda(grad_output.data_ptr<float>(), x.data_ptr<float>(), alpha.item<float>(), grad_x.data_ptr<float>(), x.numel());
        // TODO: 报错
        return grad_x;
    }
    
}

class atan_atf: public Function<atan_atf>
{
    public:
  static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor & x, const torch::Tensor & alpha)
  {
      if (x.requires_grad())
      {
        ctx->save_for_backward({x, alpha});
      }
      return x.ge(0).to(x);
  }
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
  {
      auto saved = ctx->get_saved_variables();
      auto x = saved[0];
      auto alpha = saved[1];
      auto grad_x = atan_backward(grad_outputs[0], x, alpha);
      return {grad_x, torch::Tensor()};
  }
};

torch::Tensor atan_apply(const torch::Tensor & x, const torch::Tensor & alpha)
{
    return atan_atf::apply(x, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_backward", &torch::sigmoid_backward);
    m.def("alpha_sigmoid_backward", &alpha_sigmoid_backward);
    m.def("atan", &atan_apply);
}