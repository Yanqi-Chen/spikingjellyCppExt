#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__forceinline__  __device__ float grad_atan(const float & alpha, const float & x)
{
  const float M_PI_2__alpha__x = M_PI_2 * alpha * x;
  return alpha / 2.0f / (1.0f + M_PI_2__alpha__x * M_PI_2__alpha__x);
}

__forceinline__  __device__ float grad_sigmoid(const float & alpha, const float & x)
{
  const float sigmoid_ax = 1.0f / (1.0f + expf(- alpha * x));
  return (1 - sigmoid_ax) * sigmoid_ax * alpha;
}

typedef float (*grad_surrogate_function) (const float &, const float &);

__device__ const grad_surrogate_function grad_surrogate_function_pointer[2] = { 
    grad_atan, 
    grad_sigmoid
    };


__global__ void LIF_hard_reset_backward_cuda_kernel(
    float* __restrict__ grad_x, float* __restrict__ grad_v,
    const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next,
    const float* __restrict__ x,  const float* __restrict__ h,  const float* __restrict__ spike, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau, const float alpha, const bool detach_reset,
    const int grad_surrogate_function_index
    )
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const float grad_spike_to_h = grad_surrogate_function_pointer[grad_surrogate_function_index](alpha, h[index] - v_th);
    float grad_h;
    if (detach_reset)
    {
      grad_h = grad_spike[index] * grad_spike_to_h + grad_v_next[index] * (1 - spike[index]);
    }
    else
    {
      grad_h = grad_spike[index] * grad_spike_to_h + grad_v_next[index] * (1 - spike[index] + (v_reset - h[index]) * grad_spike_to_h);
    }
    grad_x[index] = grad_h * reciprocal_tau;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
  }
}

void LIF_hard_reset_backward_cuda(
  float* grad_x, float* grad_v,
  const float* grad_spike, const float* grad_v_next,
  const float* x, const float* h, const float* spike, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & tau, const float & alpha, const bool & detach_reset, const int & grad_surrogate_function_index)
{
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    cudaError_t error = cudaSetDevice(gpu_id);
    if(error != cudaSuccess) 
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    const float reciprocal_tau = 1 / tau;
    LIF_hard_reset_backward_cuda_kernel<<<blocks, threads>>>(
      grad_x, grad_v, grad_spike, grad_v_next, 
      x, h, spike, 
      v_th, v_reset, size, 
      reciprocal_tau, 1 - reciprocal_tau, alpha, detach_reset, 
      grad_surrogate_function_index
    );
}