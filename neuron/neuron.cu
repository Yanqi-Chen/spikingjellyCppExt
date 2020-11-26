#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "neuron_def.h"
//LIF--------------------------------------------
DEF_HARD_RESET_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  HARD_RESET_KERNEL(one_sub_reciprocal_tau * v[index] + reciprocal_tau * (x[index] + v_reset));
}
DEF_HARD_RESET_FORWARD_CUDA_FUNCTION(LIF_hard_reset_forward_cuda, const float & tau)
{
  INIT_THREAD_DEVICE;
  const float reciprocal_tau = 1 / tau;
  CALL_HARD_RESET_KERNEL_FUNCTION(LIF_hard_reset_forward_cuda_kernel, reciprocal_tau, 1 - reciprocal_tau);
}


DEF_SOFT_RESET_KERNEL_FUNCTION(LIF_soft_reset_forward_cuda_kernel, const float reciprocal_tau, const float one_sub_reciprocal_tau)
{
  SOFT_RESET_KERNEL(one_sub_reciprocal_tau * v[index] + reciprocal_tau * x[index]);
}
DEF_SOFT_RESET_FORWARD_CUDA_FUNCTION(LIF_soft_reset_forward_cuda, const float & tau)
{
  INIT_THREAD_DEVICE;
  const float reciprocal_tau = 1 / tau;
  CALL_SOFT_RESET_KERNEL_FUNCTION(LIF_soft_reset_forward_cuda_kernel, reciprocal_tau, 1 - reciprocal_tau);
}