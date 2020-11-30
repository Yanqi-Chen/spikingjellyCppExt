#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void LIF_hard_reset_forward_cuda_kernel(
    const float* __restrict__ x, const float* __restrict__ v,  float* __restrict__ h,  float* __restrict__ spike, float* __restrict__ v_next, 
    const float v_th, const float v_reset, const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau
    )
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    h[index] = one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset);
    if (h[index] >= v_th)
    {
      spike[index] = 1.0f;
      v_next[index] = v_reset;
    }
    else 
    {
      spike[index] = 0.0f;
      v_next[index] = h[index];
    }
  }
}

void LIF_hard_reset_forward_cuda(const float* x, const float* v, float* h, float* spike, float* v_next, 
  const float & v_th, const float & v_reset, const int & size, const int & gpu_id, 
  const float & tau)
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
    LIF_hard_reset_forward_cuda_kernel<<<blocks, threads>>>(x, v, h, spike, v_next, v_th, v_reset, size, reciprocal_tau, 1 - reciprocal_tau);
}