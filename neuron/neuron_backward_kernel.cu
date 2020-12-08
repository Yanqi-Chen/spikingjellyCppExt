#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"

__global__ void LIF_hard_reset_backward_cuda_kernel(
    float* __restrict__ grad_x, float* __restrict__ grad_v,
    const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
    const int size,
    const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
  {
    const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
    grad_x[index] = grad_h * reciprocal_tau;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
  }
}

void LIF_hard_reset_backward_cuda(
  float* grad_x, float* grad_v,
  const float* grad_spike, const float* grad_v_next, const float* grad_s_to_h, const float* grad_v_to_h,
  const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  LIF_hard_reset_backward_cuda_kernel<<<blocks, threads>>>(
    grad_x, grad_v, grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h,
    size, 
    reciprocal_tau, 1 - reciprocal_tau
  );
}

__global__ void ParametricLIF_hard_reset_backward_cuda_kernel(
  float* __restrict__ grad_x, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike, const float* __restrict__ grad_v_next, 
  const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < size)
  {
    const float grad_h = grad_spike[index] * grad_s_to_h[index] + grad_v_next[index] * grad_v_to_h[index];
    grad_x[index] = grad_h * reciprocal_tau;
    grad_v[index] = grad_h * one_sub_reciprocal_tau;
    sdata[threadIdx.x] = grad_h * grad_h_to_rtau[index];
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x==0)
  {
    grad_rtau[0] = sdata[0];
  }
}

void ParametricLIF_hard_reset_backward_cuda(
  float* grad_x, float* grad_v, float* grad_rtau,
  const float* grad_spike, const float* grad_v_next, const float* grad_s_to_h, const float* grad_v_to_h, const float* grad_h_to_rtau,
  const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  const int threads = THREADS;
  const int blocks = (size + threads - 1) / threads;
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  ParametricLIF_hard_reset_backward_cuda_kernel<<<blocks, threads>>>(
    grad_x, grad_v, grad_rtau, 
    grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h, grad_h_to_rtau,
    size, 
    reciprocal_tau, 1 - reciprocal_tau
  );
}
//bptt-----------------------------------------

__global__ void LIF_hard_reset_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h,
  const int neuron_num, const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < neuron_num)
  {
    float grad_h;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
      grad_x_seq[mem_index] = grad_h * reciprocal_tau;
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
    }
  }
}

void LIF_hard_reset_bptt_cuda(
  float* grad_x_seq, float* grad_v, 
  const float* grad_spike_seq, const float* grad_s_to_h, const float* grad_v_to_h,
  const int & seq_len, const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  LIF_hard_reset_bptt_cuda_kernel<<<blocks, threads>>>(
    grad_x_seq, grad_v,
    grad_spike_seq, grad_s_to_h, grad_v_to_h, 
    neuron_num, size,
    reciprocal_tau, 1 - reciprocal_tau
  );
}

__global__ void ParametricLIF_hard_reset_bptt_cuda_kernel(
  float* __restrict__ grad_x_seq, float* __restrict__ grad_v, float* __restrict__ grad_rtau,
  const float* __restrict__ grad_spike_seq, const float* __restrict__ grad_s_to_h, const float* __restrict__ grad_v_to_h, const float* __restrict__ grad_h_to_rtau,
  const int neuron_num, const int size,
  const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sdata[THREADS];
  if (index < neuron_num)
  {
    float grad_h;
    float sum_t_grad_h_to_rtau = 0.0f;
    for(int mem_offset = size - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
    {
      const int mem_index = index + mem_offset;
      grad_h = grad_spike_seq[mem_index] * grad_s_to_h[mem_index] + grad_v[index] * grad_v_to_h[mem_index];
      grad_x_seq[mem_index] = grad_h * reciprocal_tau;
      grad_v[index] = grad_h * one_sub_reciprocal_tau;
      sum_t_grad_h_to_rtau += grad_h * grad_h_to_rtau[mem_index];
    }
    sdata[threadIdx.x] = sum_t_grad_h_to_rtau;
  }
  else
  {
    sdata[threadIdx.x] = 0.0f;
  }
  int threadx = blockDim.x;
  #pragma unroll
  for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
  {
    // Synchronize all thread before next loop
    __syncthreads();
    if (threadIdx.x < stride)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x==0)
  {
    grad_rtau[0] = sdata[0];
  }
}

void ParametricLIF_hard_reset_bptt_cuda(
  float* grad_x_seq, float* grad_v, float* grad_rtau,
  const float* grad_spike_seq, const float* grad_s_to_h, const float* grad_v_to_h, const float* grad_h_to_rtau,
  const int & seq_len, const int & size, const int & gpu_id, 
  const float & reciprocal_tau)
{
  CHECK_CUDA_OPERATION(cudaSetDevice(gpu_id));
  const int threads = THREADS;
  const int neuron_num = size / seq_len;
  const int blocks = (neuron_num + threads - 1) / threads;
  ParametricLIF_hard_reset_bptt_cuda_kernel<<<blocks, threads>>>(
    grad_x_seq, grad_v, grad_rtau,
    grad_spike_seq, grad_s_to_h, grad_v_to_h, grad_h_to_rtau,
    neuron_num, size,
    reciprocal_tau, 1 - reciprocal_tau
  );
}