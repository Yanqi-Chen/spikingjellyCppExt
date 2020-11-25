#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <math.h>
#include <stdio.h>

// 确定线程数量，设置GPU的代码段
#define INIT_THREAD_DEVICE const int threads = 1024;const int blocks = (size + threads - 1) / threads;cudaSetDevice(gpu_id);

// hard reset的代码段
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

// soft reset的代码段
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

// hard reset的核函数代码段。用户只需要定义如何充电
#define HARD_RESET_KERNEL_FUNCTION(charge_function) do { \
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    const float h = charge_function; \
    HARD_RESET; \
  } \
} while(0)

// 定义核函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_KERNEL_FUNCTION(function_name, ...) __global__ void function_name (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ s, float* __restrict__ v_next, const float v_th, const float v_reset, const int size, __VA_ARGS__)

// 调用核函数的代码段。function_name是核函数的名字，...是额外的参数
#define CALL_KERNEL_FUNCTION(function_name, ...) function_name<<<blocks, threads>>>(x, v, s, v_next, v_th, v_reset, size, __VA_ARGS__);

// 定义
#define DEF_FORWARD_FUNCTION(function_name, ...) void function_name (const float* x, const float* v, float* s, float* v_next, const float & v_th, const float & v_reset, const int & size, const int & gpu_id, __VA_ARGS__)

//LIF--------------------------------------------
DEF_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  HARD_RESET_KERNEL_FUNCTION(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}

DEF_FORWARD_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau)
{
  INIT_THREAD_DEVICE;
  const float reciprocal_tau = 1 / tau;
  CALL_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, reciprocal_tau, 1 - reciprocal_tau);
}
