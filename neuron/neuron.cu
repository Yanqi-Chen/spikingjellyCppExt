#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

//LIF--------------------------------------------
__global__ void LIF_hard_reset_forward_cuda_kernel (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ s, float* __restrict__ v_next, 
  const float v_th, const float v_reset, const int size, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
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
void LIF_hard_reset_forward_cuda (const float* x, const float* v, float* s, float* v_next, const float & v_th, 
  const float & v_reset, const int & size, const int & gpu_id, const float & tau)
{
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  cudaSetDevice(gpu_id);
  const float reciprocal_tau = 1 / tau;
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, s, v_next, v_th, v_reset, size, reciprocal_tau, 1 - reciprocal_tau);
}
