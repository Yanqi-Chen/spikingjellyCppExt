#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>

#define HARD_RESET do { \
  if (h >= v_th) \
  { \
    s[index] = 1.0f; \
    v_next[index] = v_reset; \
  } \
  else \
  { \
    s[index] = 0.0f; \
    v_next[index] = h; \
  } \
} while(0)

#define SOFT_RESET do { \
  if (h >= v_th) \
  { \
    s[index] = 1.0f; \
    v_next[index] = v[index] - v_th; \
  } \
  else \
  { \
    s[index] = 0.0f; \
    v_next[index] = h; \
  } \
} while(0)


#define HARD_RESET_KERNEL_FUNCTION(charge_function) do { \
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    const float h = charge_function; \
    HARD_RESET; \
  } \
} while(0)

#define DEF_KERNEL_FUNCTION(function_name, ...) __global__ void function_name (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ v_next, float* __restrict__ s, const float v_th, const float v_reset, const int size, __VA_ARGS__)

#define

//LIF--------------------------------------------
DEF_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  HARD_RESET_KERNEL_FUNCTION(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* v_next, float* s, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, const float & tau)
{
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  cudaSetDevice(gpu_id);

  const float reciprocal_tau = 1 / tau;
  LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, v_next, s, v_th, v_reset, size, reciprocal_tau, 1 - reciprocal_tau);
}