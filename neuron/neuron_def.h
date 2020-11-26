#pragma once

// cpp函数相关的定义
// 定义HARD_RESET的前向传播函数。function_name是函数的名字，...是额外的参数
#define DEF_HARD_RESET_FORWARD_FUNCTION(function_name, ...) std::vector<at::Tensor> function_name(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset, __VA_ARGS__)




// cuda函数相关的定义--------------------------------
// 确定线程数量，设置GPU的代码段
#define INIT_THREAD_DEVICE const int threads = 1024;const int blocks = (size + threads - 1) / threads;cudaError_t error = cudaGetLastError();if(error != cudaSuccess) {printf("CUDA error: %s\n", cudaGetErrorString(error));exit(-1);}


// hard reset---------------------------
// 定义cuda前向传播函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(function_name, ...) void function_name (const float* x, const float* v, float* spike, float* v_next, const float & v_th, const float & v_reset, const int & size, const int & gpu_id, __VA_ARGS__)


// hard reset的代码段
#define HARD_RESET do { \
  if (h >= v_th) \
  { \
    spike[index] = 1.0f; \
    v_next[index] = v_reset; \
  } \
  else \
  { \
    spike[index] = 0.0f; \
    v_next[index] = h; \
  } \
} while(0)

// hard reset的核函数代码段。用户只需要定义如何充电
#define HARD_RESET_KERNEL_FUNCTION(charge_function) do {\
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    const float h = charge_function; \
    HARD_RESET; \
  } \
} while(0)

// 定义核函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_HARD_RESET_KERNEL_FUNCTION(function_name, ...) __global__ void function_name (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next, const float v_th, const float v_reset, const int size, __VA_ARGS__)

// 调用核函数的代码段。function_name是核函数的名字，...是额外的参数
#define CALL_HARD_RESET_KERNEL_FUNCTION(function_name, ...) function_name<<<blocks, threads>>>(x, v, spike, v_next, v_th, v_reset, size, __VA_ARGS__)

// soft reset---------------------------
// soft reset的代码段
#define SOFT_RESET do {\
  if (h >= v_th) \
  { \
    spike[index] = 1.0f; \
    v_next[index] = v[index] - v_th; \
  } \
  else \
  { \
    spike[index] = 0.0f; \
    v_next[index] = h; \
  } \
} while(0)