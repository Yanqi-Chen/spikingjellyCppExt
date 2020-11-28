#pragma once

// cpp函数相关的定义

// 定义hard reset的前向传播函数。function_name是函数的名字，...是额外的参数
#define DEF_HARD_RESET_FORWARD_FUNCTION(function_name, ...) std::vector<at::Tensor> function_name(torch::Tensor & x, torch::Tensor & v, const float & v_th, const float & v_reset, ##__VA_ARGS__)
// 检查输入、调用hard reset前向传播的CUDA函数。function_name是函数的名字，...是额外的参数
#define CHECK_INPUT_AND_CALL_HARD_RESET_FORWARD_CUDA_FUNCTION(function_name, ...) do { \
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor"); \
  TORCH_CHECK(v.device().is_cuda(), "v must be a CUDA tensor"); \
  if (! x.is_contiguous()) \
  { \
      x = x.contiguous(); \
  } \
  if (! v.is_contiguous()) \
  { \
      v = v.contiguous(); \
  } \
  auto v_next = torch::zeros_like(v.data()); \
  auto spike = torch::zeros_like(v.data()); \
  if (! v_next.is_contiguous()) \
  { \
      v_next = v_next.contiguous(); \
  } \
  if (! spike.is_contiguous()) \
  { \
      spike = spike.contiguous(); \
  } \
  function_name(x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), v_th, v_reset, x.numel(), x.get_device(), ##__VA_ARGS__); \
  return {spike, v_next}; \
} while(0)



// cuda函数相关的定义--------------------------------
// 确定线程数量，设置GPU的代码段
#define INIT_THREAD_DEVICE const int threads = 1024;const int blocks = (size + threads - 1) / threads;cudaError_t error = cudaSetDevice(gpu_id);if(error != cudaSuccess) {printf("CUDA error: %s\n", cudaGetErrorString(error));exit(-1);}


// hard reset---------------------------
// 定义hard reset cuda前向传播函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(function_name, ...) void function_name (const float* x, const float* v, float* spike, float* v_next, const float & v_th, const float & v_reset, const int & size, const int & gpu_id, ##__VA_ARGS__)


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
#define HARD_RESET_KERNEL(charge_function) do {\
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    const float h = charge_function; \
    HARD_RESET; \
  } \
} while(0)

// 定义核函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_HARD_RESET_KERNEL_FUNCTION(function_name, ...) __global__ void function_name (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next, const float v_th, const float v_reset, const int size, ##__VA_ARGS__)

// 调用核函数的代码段。function_name是核函数的名字，...是额外的参数
#define CALL_HARD_RESET_KERNEL_FUNCTION(function_name, ...) function_name<<<blocks, threads>>>(x, v, spike, v_next, v_th, v_reset, size, ##__VA_ARGS__)















// soft reset---------------------------
#define DEF_SOFT_RESET_FORWARD_FUNCTION(function_name, ...) std::vector<at::Tensor> function_name(torch::Tensor & x, torch::Tensor & v, const float & v_th, ##__VA_ARGS__)
// 检查输入、调用hard reset前向传播的CUDA函数。function_name是函数的名字，...是额外的参数
#define CHECK_INPUT_AND_CALL_SOFT_RESET_FORWARD_CUDA_FUNCTION(function_name, ...) do { \
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor"); \
  TORCH_CHECK(v.device().is_cuda(), "v must be a CUDA tensor"); \
  if (! x.is_contiguous()) \
  { \
      x = x.contiguous(); \
  } \
  if (! v.is_contiguous()) \
  { \
      v = v.contiguous(); \
  } \
  auto v_next = torch::zeros_like(v.data()); \
  auto spike = torch::zeros_like(v.data()); \
  if (! v_next.is_contiguous()) \
  { \
      v_next = v_next.contiguous(); \
  } \
  if (! spike.is_contiguous()) \
  { \
      spike = spike.contiguous(); \
  } \
  function_name(x.data_ptr<float>(), v.data_ptr<float>(), spike.data_ptr<float>(), v_next.data_ptr<float>(), v_th, x.numel(), x.get_device(), ##__VA_ARGS__); \
  return {spike, v_next}; \
} while(0)
// 定义soft reset cuda前向传播函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_SOFT_RESET_FORWARD_CUDA_FUNCTION(function_name, ...) void function_name (const float* x, const float* v, float* spike, float* v_next, const float & v_th, const int & size, const int & gpu_id, ##__VA_ARGS__)

// 定义核函数的代码段。function_name是核函数的名字，...是额外的参数
#define DEF_SOFT_RESET_KERNEL_FUNCTION(function_name, ...) __global__ void function_name (const float* __restrict__ x, const float* __restrict__ v, float* __restrict__ spike, float* __restrict__ v_next, const float v_th, const int size, ##__VA_ARGS__)

// 调用核函数的代码段。function_name是核函数的名字，...是额外的参数
#define CALL_SOFT_RESET_KERNEL_FUNCTION(function_name, ...) function_name<<<blocks, threads>>>(x, v, spike, v_next, v_th, size, ##__VA_ARGS__)

// soft reset的核函数代码段。用户只需要定义如何充电
#define SOFT_RESET_KERNEL(charge_function) do {\
  const int index = blockIdx.x * blockDim.x + threadIdx.x; \
  if (index < size) \
  { \
    const float h = charge_function; \
    SOFT_RESET; \
  } \
} while(0)

// soft reset的代码段
#define SOFT_RESET do { \
  if (h >= v_th) \
  { \
    spike[index] = 1.0f; \
    v_next[index] = h - v_th; \
  } \
  else \
  { \
    spike[index] = 0.0f; \
    v_next[index] = h; \
  } \
} while(0)