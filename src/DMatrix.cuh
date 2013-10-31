#ifndef DMATRIX_CUH
#define DMATRIX_CUH
#include <common.cuh>
#include <cstdio>
#include <cublas_v2.h>


template<class T> 
class DMatrix {
    cublasStatus_t _handle;
    int _nrows, int _ncols;
    T* _host_data;
    T* _dev_data;
public:
    DMatrix(cublasStatus_t handle, nrows, ncols, T* data = 0);
    
    
};


#endif //DMATRIX_CUH
