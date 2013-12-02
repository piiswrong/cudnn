#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <assert.h>

#include <mkl.h>

const int SPARSE_DEGREE = 15;
#ifndef DISABLE_GPU

#include <cublas_v2.h>
#include <curand.h>
#include <math_functions.h>

#define HOSTDEVICE __host__ __device__
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)
#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

cublasStatus_t  cublasXasum(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result);

cublasStatus_t  cublasXasum(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result);							

cublasStatus_t  cublasXnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result);

cublasStatus_t  cublasXnrm2(cublasHandle_t handle, int n,
                            const double          *x, int incx, double *result);

cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const float           *alpha,
                                        const float           *x, int incx,
                                        float                 *y, int incy);

cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                                        const double          *alpha,
                                        const double          *x, int incx,
                                        double                *y, int incy);

cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const float *alpha, float *A, int lda,
                                        float *B, int ldb, const float *beta,
                                        float *C, int ldc);

cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                        cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                        const double *alpha, double *A, int lda,
                                        double *B, int ldb, const double *beta,
                                        double *C, int ldc);
#else
typedef void* cublasHandle_t; 
typedef void* curandState;
typedef void* cudaStream_t;
#define CUDA_CALL(x) 
#define CUBLAS_CALL(x) 
#define CURAND_CALL(x)
#define HOSTDEVICE 
#include <math.h>
using namespace std;
#endif

#ifdef ADMM
#define USE_MPI
#endif

#ifdef USE_MPI
#include <mpi.h>
extern int mpi_world_size;
extern int mpi_world_rank;
#endif


float cblas_Xasum (const MKL_INT N, const float *X, const MKL_INT incX);
double cblas_Xasum (const MKL_INT N, const double *X, const MKL_INT incX);
void cblas_Xaxpy (const MKL_INT N, const float alpha, const float *X, const MKL_INT incX, float *Y, const MKL_INT incY);
void cblas_Xaxpy (const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, double *Y, const MKL_INT incY);
float cblas_Xnrm2 (const MKL_INT N, const float *X, const MKL_INT incX);
double cblas_Xnrm2 (const MKL_INT N, const double *X, const MKL_INT incX);
void cblas_Xgemm (const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const float alpha, const float *A, const MKL_INT lda, const float *B, const MKL_INT ldb, const float beta, float *C, const MKL_INT ldc);
void cblas_Xgemm (const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N, const MKL_INT K, const double alpha, const double *A, const MKL_INT lda, const double *B, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc);

#endif //COMMON_H
