#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#define BLOCK_SIZE 32

namespace {
    template <typename scalar_t>
    __global__ void spike_matmul_cuda_forward_kernel(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C
    )
    {
        //https://github.com/NVIDIA/cuda-samples/blob/master/Samples/matrixMul/matrixMul.cu
        int wA = A.size(1);
        int wB = B.size(1);

        // Block index
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        
        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd   = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep  = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep  = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            As[ty][tx] = A[ty][a + tx];
            Bs[ty][tx] = B[ty][b + tx];

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
        #pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if (As[ty][k] == 1)
                {
                    Csub += Bs[k][tx];
                }    
                // Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        C[ty][c + tx] = Csub;
    }
}

torch::Tensor spike_matmul_cuda_forward(const torch::Tensor & A, const torch::Tensor & B, torch::Tensor & C)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(B.size(1) / threads.x, A.size(0) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(B.type(), "spike_matmul_cuda_forward", ([&] {
        spike_matmul_cuda_forward_kernel<scalar_t> <<<grid, threads>>>(
        A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        C.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
    );
    }));
    return C;
}
