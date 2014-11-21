#ifndef COMMON_CUH
#define COMMON_CUH
#ifndef DISABLE_GPU

#include <common.h>

#include <cublas_v2.h>
#include <curand.h>
#include <math_functions.h>

#define HOSTDEVICE __host__ __device__
const int WARP_SIZE = 16;
const int BLOCK_SIZE = 256;
const int TILE_DIM = 16;
const int BLOCK_ROWS = 8;

#define CUDA_CALL(statment) do {   cudaError_t macroErrorCode = statment; \
                            if((macroErrorCode) != cudaSuccess) { \
                            printf("Cuda Error at %s:%d with code %d(!=%d): %s\n",__FILE__,__LINE__, macroErrorCode, cudaSuccess, cudaGetErrorString(macroErrorCode));\
                            assert(false);\
                            exit(EXIT_FAILURE);}} while(0)
#define CUBLAS_CALL(statment) do { int macroErrorCode = statment; \
                            if((macroErrorCode) != CUBLAS_STATUS_SUCCESS) { \
                            printf("Cublas Error at %s:%d with code %d(!=%d): %s\n",__FILE__,__LINE__, macroErrorCode, CUBLAS_STATUS_SUCCESS, cudaGetErrorString((cudaError_t)macroErrorCode));\
                            assert(false);\
                            exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(statment) do { int macroErrorCode = statment; \
                            if((macroErrorCode) != CURAND_STATUS_SUCCESS) { \
                            printf("Curand Error at %s:%d with code %d(!=%d)\n",__FILE__,__LINE__, macroErrorCode, CURAND_STATUS_SUCCESS);\
                            assert(false);\
                            exit(EXIT_FAILURE);}} while(0)
#ifdef NDEBUG
#define CUDA_KERNEL_CHECK() do {  \
                            CUDA_CALL(cudaPeekAtLastError()); \
                            }while(0)
#else
#define CUDA_KERNEL_CHECK() do { \
                            CUDA_CALL(cudaPeekAtLastError()); \
                            CUDA_CALL(cudaDeviceSynchronize()); \
                            CUDA_CALL(cudaPeekAtLastError()); \
                            }while(0)
#endif
                            

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

cublasStatus_t cublasXdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                        int m, int n,
                                        const float           *A, int lda,
                                        const float           *x, int incx,
                                        float           *C, int ldc);

cublasStatus_t cublasXdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                                        int m, int n,
                                        const double          *A, int lda,
                                        const double          *x, int incx,
                                        double          *C, int ldc);

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                                        const float           *x, int incx,
                                        const float           *y, int incy,
                                        float           *result);

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                                        const double          *x, int incx,
                                        const double          *y, int incy,
                                        double          *result);
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

#endif //COMMON_H
