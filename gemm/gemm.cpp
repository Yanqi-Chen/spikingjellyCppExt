#include <iostream>
#include <torch/extension.h>

void sparse_mm_dense_cusparse_backend(const int & cuda_device_id, const int & m, const int & n, const int & p, float * dA, float * dB, float * dC);

void sparse_mm_dense_cusparse(const torch::Tensor & A, const torch::Tensor & B, torch::Tensor & C)
{   
    // A is sparse, B is dense
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
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