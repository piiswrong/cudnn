#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <assert.h>

#include <mkl.h>

#include <cudnn.h>

#include <fenv.h>


const int SPARSE_DEGREE = 15;
const int VERBOSE_NONE = 0;
const int VERBOSE_MINIMAL = 1;
const int VERBOSE_NORMAL = 2;
extern FILE* flog;
extern int log_verbosity;
#define LOG(v, x) \
if (flog != NULL && v <= log_verbosity) { \
    x; \
    fflush(flog); \
}

#ifdef ADMM
#define USE_MPI
#endif

#ifdef DOWN_POUR_SGD
#define USE_MPI
extern int sgd_num_param_server;
const int SGD_LOSS_TAG = 1024;
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

std::vector<std::string> split(const std::string &str, const std::string &delimiters);
    
class DDim4 {
public:
    int n,c,h,w;
    DDim4() {}
    DDim4(int _n, int _c, int _h, int _w) : n(_n), c(_c), h(_h), w(_w) {}
};


#endif //COMMON_H
