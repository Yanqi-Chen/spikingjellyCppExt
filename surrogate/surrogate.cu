#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
__global__ void alpha_atan_backward_cuda_kernel(const float* __restrict__ grad_output, const float* __restrict__ x, const float alpha,
    float* __restrict__ grad_x, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      grad_x[index] = alpha / 2.0f / (1.0f + powf(M_PI_2 * alpha * x[index], 2)) * grad_output[index];
  }
}

void alpha_atan_backward_cuda(const float* grad_output, const float* x, const float & alpha, float* grad_x, const int & size)
{
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  alpha_atan_backward_cuda_kernel<<<blocks, threads>>>(grad_output, x, alpha, grad_x, size);
}