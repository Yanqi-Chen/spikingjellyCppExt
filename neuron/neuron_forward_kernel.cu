#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"
#define HARD_RESET_FORWARD_CUDA_KERNEL(charge_function) do { \
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    h[index] = charge_function; \
    if (h[index] >= v_th) \
    { \
      spike[index] = 1.0f; \
      v_next[index] = v_reset; \
    } \
    else \
    { \
      spike[index] = 0.0f; \
      v_next[index] = h[index]; \
    } \
  } \
} while(0) \

__global__ void LIF_hard_reset_forward_cuda_kernel(
    const float* __restrict__ x, const float* __restrict__ v,  float* __restrict__ h,  float* __restrict__ spike, float* __restrict__ v_next, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau
    )
{
  HARD_RESET_FORWARD_CUDA_KERNEL(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* h, float* spike, float* v_next, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & tau)
{
  INIT_DEVIDE_THREAD;
  const float reciprocal_tau = 1 / tau;
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, h, spike, v_next, v_th, v_reset, size, reciprocal_tau, 1 - reciprocal_tau);
}