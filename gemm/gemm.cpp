#include <iostream>
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void sparse_mm_dense_cusparse_backend(const int & cuda_device_id, const int & m, const int & n, const int & p, float * dA, float * dB, float * dC);

void sparse_mm_dense_cusparse(const torch::Tensor & A, const torch::Tensor & B, torch::Tensor & C)
{   
    // A is sparse, B is dense
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    // A: [M, N] B:[N, P]
    // C: [M, P]
    // Mat size. In cuSparse, dense matrix is column-major format while sparse matrix is row-major.So we use csc instead of csr
    int m = A.size(0);
    int n = A.size(1);
    int p = B.size(1);
    float* dA = (float*)A.data_ptr<float>();
    float* dB = (float*)B.data_ptr<float>();
    float* dC = (float*)C.data_ptr<float>();
    sparse_mm_dense_cusparse_backend(B.get_device(), m, n, p, dA, dB, dC);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_mm_dense_cusparse", &sparse_mm_dense_cusparse);
}