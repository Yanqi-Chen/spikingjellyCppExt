#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__forceinline__  __device__ float grad_atan(const float & alpha, const float & x)
{
  printf("%f %f ",M_PI_2 * alpha * x, powf(M_PI_2 * alpha * x, 2));
  return alpha / 2.0f / (1.0f + powf(M_PI_2 * alpha * x, 2));
  
}

__global__ void LIF_hard_reset_backward_cuda_kernel_atan(
    float* __restrict__ grad_x, float* __restrict__ grad_v,
    const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next,
    const float* __restrict__ x,  const float* __restrict__ h,  const float* __restrict__ spike, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau, const float alpha, const bool detach_reset
    )
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    float grad_spike_to_h = grad_atan(alpha, h[index] - v_th);
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

void LIF_hard_reset_backward_cuda_atan(
  float* grad_x, float* grad_v,
  const float* grad_spike, const float* grad_v_next,
  const float* x, const float* h, const float* spike, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & tau, const float & alpha, const bool & detach_reset)
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
    LIF_hard_reset_backward_cuda_kernel_atan<<<blocks, threads>>>(
      grad_x, grad_v, grad_spike, grad_v_next, 
      x, h, spike, 
      v_th, v_reset, size, 
      reciprocal_tau, 1 - reciprocal_tau, alpha, detach_reset);
}