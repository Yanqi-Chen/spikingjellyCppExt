#include <iostream>
#include <torch/extension.h>
using namespace torch::autograd;
torch::Tensor heaviside_step(const torch::Tensor & x)
{   
    return x.ge(0).to(x.dtype());  // bool -> float
}

class sigmoid: public Function<sigmoid>
{
    public:
  static torch::Tensor forward(AutogradContext *ctx, const torch::Tensor & x, const double & alpha)
  {
      if (x.requires_grad())
        ctx->saved_data["alpha"] = alpha;
        ctx->save_for_backward({x});
      return heaviside_step(x);
  }
  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
  {
      auto alpha = ctx->saved_data["alpha"].toDouble();
      auto saved = ctx->get_saved_variables();
      auto x = saved[0];
      auto grad_x = alpha * torch::sigmoid_backward(grad_outputs[0], torch::sigmoid(x * alpha));  // main contribution to acceleration
      return {grad_x, torch::Tensor()};
  }
};

torch::Tensor sigmoid_apply(const torch::Tensor & x, const double & alpha)
{
    return sigmoid::apply(x, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_apply", sigmoid_apply);
}