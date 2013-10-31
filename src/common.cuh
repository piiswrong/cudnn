#ifndef COMMON_H
#define COMMON_H

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \                                                                
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                     
#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \                                                                
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                     
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \               
                            printf("Error at %s:%d\n",__FILE__,__LINE__);\     
                            exit(EXIT_FAILURE);}} while(0)                       

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
