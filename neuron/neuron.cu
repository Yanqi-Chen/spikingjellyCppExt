#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>
//LIF--------------------------------------------
__global__ void LIF_hard_reset_forward_cuda_kernel(const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ v_next, float* __restrict__ s, 
  const float v_th, const float v_reset, const float reciprocal_tau, const float one_sub_reciprocal_tau, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) 
  {
    const float h = one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset);
    if (h >= v_th)
    {
      s[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else
    {
      s[index] = 0.0f;
      v_next[index] = h;
    }
  }
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* v_next, float* s, 
  const float & v_th, const float & v_reset, const float & tau, const int & size)
{
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    const float reciprocal_tau = 1 / tau;
    LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, v_next, s, v_th, v_reset, reciprocal_tau, 1 - reciprocal_tau, size);
}



__global__ void LIF_soft_reset_forward_cuda_kernel(const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ v_next, float* __restrict__ s, 
  const float v_th, const float reciprocal_tau, const float one_sub_reciprocal_tau, const int size)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float h = one_sub_reciprocal_tau * v[index] + reciprocal_tau * x[index];
    if (h >= v_th)
    {
      s[index] = 1.0f;
      v_next[index] = v[index] - v_th;
    }
    else
    {
      s[index] = 0.0f;
      v_next[index] = h;
    }
  }
}

void LIF_soft_reset_forward_cuda(const float* x, const float* v, float* v_next, float* s, 
  const float & v_th, const float & tau, const int & size)
{
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    const float reciprocal_tau = 1 / tau;
    LIF_soft_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, v_next, s, v_th, reciprocal_tau, 1 - reciprocal_tau, size);
}