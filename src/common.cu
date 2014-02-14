#include <common.cuh>

FILE* flog = NULL;

#ifdef DOWN_POUR_SGD
int sgd_num_param_server = 1;
#endif

#ifdef USE_MPI
int mpi_world_size;
int mpi_world_rank;
#endif

#ifndef DISABLE_GPU

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

cublasStatus_t cublasXdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                        int m, int n,
                                        const float           *A, int lda,
                                        const float           *x, int incx,
                                        float           *C, int ldc)
{
    return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t cublasXdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                        int m, int n,
                                        const double          *A, int lda,
                                        const double          *x, int incx,
                                        double          *C, int ldc)
{
    return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                                        const float           *x, int incx,
                                        const float           *y, int incy,
                                        float           *result) 
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                                        const double          *x, int incx,
                                        const double          *y, int incy,
                                        double          *result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}
#endif

float cblas_Xasum (const MKL_INT N, const float *X, const MKL_INT incX) {
    return cblas_sasum (N, X, incX); 
}

double cblas_Xasum (const MKL_INT N, const double *X, const MKL_INT incX) {
    return cblas_dasum (N, X, incX);
}

void cblas_Xaxpy (const MKL_INT N, const float alpha, const float *X, const MKL_INT incX, float *Y, const MKL_INT incY) {
    cblas_saxpy (N, alpha, X, incX, Y, incY);
}
void cblas_Xaxpy (const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, double *Y, const MKL_INT incY) {
    cblas_daxpy (N, alpha, X, incX, Y, incY);
}

float cblas_Xnrm2 (const MKL_INT N, const float *X, const MKL_INT incX) {
    return cblas_snrm2 (N, X, incX);
}

double cblas_Xnrm2 (const MKL_INT N, const double *X, const MKL_INT incX) {
    return cblas_dnrm2 (N, X, incX);
}

void cblas_Xgemm (const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha, const float *A, const MKL_INT lda, const float *B, const MKL_INT ldb, const float beta, float *C, const MKL_INT ldc) {
    cblas_sgemm (Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblas_Xgemm (const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha, const double *A, const MKL_INT lda, const double *B, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc) {
    cblas_dgemm (Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}



