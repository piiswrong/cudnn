#include <common.cuh>

cublasStatus_t  cublasXasum(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
{
    return cublasSasum(handle, n, x, incx, result);
}

cublasStatus_t  cublasXasum(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result)
{
    return cublasDasum(handle, n, x, incx, result);
}					
							
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
cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const float           *alpha,
                                        const float           *x, int incx,
                                        float                 *y, int incy) 
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const double          *alpha,
                                        const double          *x, int incx,
                                        double                *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const float *alpha, float *A, int lda,
                                        float *B, int ldb, const float *beta,
                                        float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const double *alpha, double *A, int lda,
                                        double *B, int ldb, const double *beta,
                                        double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


