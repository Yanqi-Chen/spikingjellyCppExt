#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>

__global__ void mul_cuda_kernel(const bool* __restrict__ spike, const float* __restrict__ x, float* __restrict__ y, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
      if (!spike[index])
      {
        y[index] = 0.0f;
      }
  }
}

void mul_cuda(const bool* spike, const float* x, float *y, const int & size, const int & gpu_id)
{
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  cudaSetDevice(gpu_id);
  mul_cuda_kernel<<<blocks, threads>>>(spike, x, y, size);
}
