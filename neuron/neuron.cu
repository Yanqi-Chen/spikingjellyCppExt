#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"
//LIF--------------------------------------------
DEF_HARD_RESET_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  HARD_RESET_KERNEL_FUNCTION(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}
DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau)
{
  //INIT_THREAD_DEVICE;
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) 
  {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  const float reciprocal_tau = 1 / tau;
  CALL_HARD_RESET_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, reciprocal_tau, 1 - reciprocal_tau);
}
