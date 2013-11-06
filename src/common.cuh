#ifndef COMMON_H
#define COMMON_H

#include <cublas_v2.h>
#include <algorithm>

const int BLOCK_SIZE = 256;
const int SPARSE_DEGREE = 15;


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \                                                                
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                     
#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \                                                                
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                     
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \               
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                       

						
							
cublasStatus_t  cublasXnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t  cublasXnrm2(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}
static inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const float           *alpha,
                                        const float           *x, int incx,
                                        float                 *y, int incy) 
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

static inline cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const double          *alpha,
                                        const double          *x, int incx,
                                        double                *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const float *alpha, float *A, int lda,
                                        float *B, int ldb, const float *beta,
                                        float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const double *alpha, double *A, int lda,
                                        double *B, int ldb, const double *beta,
                                        double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif //COMMON_H
