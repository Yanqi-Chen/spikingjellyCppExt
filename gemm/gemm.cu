#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <cublas_v2.h>
using namespace std;

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
    fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
    exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

template<class T>
struct reCuBuffer
{
T* data = NULL;
int len = 0;
};

template<class T>
void resize(reCuBuffer<T>& buffer, int size)
{
if(size > buffer.len)
{
if(buffer.len > 0)CUDA_CALL( cudaFree(buffer.data));
CUDA_CALL( cudaMalloc( &(buffer.data), size));
buffer.len = size;
}
}

static reCuBuffer<int>   nnzPerCol_, ColInd_, RowPtr_;
static reCuBuffer<float> csrVal_, tranBuffer_;

void sparse_mm_dense_cusparse_backend(const int & m, const int & n, const int & p, float * dA, float * dB, float * dC)
{   
    // CT = A * BT
    resize(tranBuffer_, m * p * sizeof(float));

    //view_cuda_tensor(A);
    //view_cuda_tensor(B);

    cusparseHandle_t  handle;
    CUSPARSE_CALL(cusparseCreate(&handle));
    cublasHandle_t handle2;
    CUDA_CALL(cublasCreate(&handle2));

    // transform dense A to csr
    cusparseMatDescr_t descrX;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));

    int total_nnz;
    resize(nnzPerCol_, m * sizeof(int));

    CUSPARSE_CALL(cusparseSnnz(handle, CUSPARSE_DIRECTION_COLUMN, n, m, descrX, dA, n, nnzPerCol_.data, &total_nnz));
    resize(csrVal_, total_nnz * sizeof(float));
    resize(ColInd_, total_nnz * sizeof(int));
    resize(RowPtr_, (m+1) * sizeof(int));

    CUSPARSE_CALL(cusparseSdense2csc(handle, n, m, descrX, dA, n, nnzPerCol_.data, csrVal_.data, ColInd_.data, RowPtr_.data));

    // B * C
    cusparseMatDescr_t descrA;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CALL(cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));

    float alpha = 1.0f;
    float beta  = 0.0f;
    CUSPARSE_CALL(cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,
        n,p,m,total_nnz,&alpha,descrA,csrVal_.data,RowPtr_.data, ColInd_.data,dB,p,&beta,tranBuffer_.data,n));

    // C need TRANSPOSE
    CUDA_CALL(cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_T, p, m, &alpha, tranBuffer_.data, m, &beta, tranBuffer_.data, m, dC, p));
    //view_cuda_tensor(C);

    CUSPARSE_CALL(cusparseDestroy(handle));
    CUDA_CALL(cublasDestroy(handle2));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrX));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrA));
}
