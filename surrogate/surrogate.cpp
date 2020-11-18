#include <iostream>
#include <torch/extension.h>
#include <math.h>

using namespace torch::autograd;

torch::Tensor heaviside_step(const torch::Tensor & x)
{   
    return x.ge(0).to(x.dtype());  // bool -> float
}

class sigmoid_atf: public Function<sigmoid_atf>
{
    public:
  static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor & x, const torch::Tensor & alpha)
  {
      if (x.requires_grad())
      {
        ctx->save_for_backward({x, alpha});
      }
      return heaviside_step(x);
  }
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
  {
      auto saved = ctx->get_saved_variables();
      auto x = saved[0];
      auto alpha = saved[1];
      auto grad_x = alpha * torch::sigmoid_backward(grad_outputs[0], torch::sigmoid(x * alpha));  // main contribution to acceleration
      return {grad_x, torch::Tensor()};
  }
};

torch::Tensor sigmoid_apply(const torch::Tensor & x, const torch::Tensor & alpha)
{
    return sigmoid_atf::apply(x, alpha);
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
      return heaviside_step(x);
  }
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
  {
      auto saved = ctx->get_saved_variables();
      auto x = saved[0];
      auto alpha = saved[1];
      auto grad_x = alpha / 2 / (1 + M_PI_2 * alpha * x).pow_(2) * grad_outputs[0];
      return {grad_x, torch::Tensor()};
  }
};

torch::Tensor atan_apply(const torch::Tensor & x, const torch::Tensor & alpha)
{
    return atan_atf::apply(x, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid", &sigmoid_apply);
    m.def("atan", &atan_apply);
}