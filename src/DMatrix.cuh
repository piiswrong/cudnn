#ifndef DMATRIX_CUH
#define DMATRIX_CUH
#include <common.cuh>
#include <cstdio>
#include <cublas_v2.h>
#include <memory.h>


template<class T> 
class DMatrix {
    cublasStatus_t _handle;
    int _nrows, int _ncols;
    int _size;
    bool _T;
    T* _host_data;
    T* _dev_data;
    bool _on_device;
public:
    DMatrix(nrows, ncols, cublasStatus_t handle = 0) {
        _nrows = nrows;
        _ncols = ncols;
        _size = _nrows*_ncols*sizeof(T);
        _dev_data = (T*)malloc(_nrows*_ncols*sizeof(T); 
        if (handle) {
            _on_device = true;
            _handle = handle;
            CUDA_CALL(cudaMalloc((void**)&_dev_data, _size));
        }
    }
    
    int nrows() { return _nrows; }
    int ncols() { return _ncols; }
    int T() { return _T; }
    void setT() { _T = !_T; }
    cublasOperation_t Tchar() { return _T ? CUBLAS_OP_T : CUBLAS_OP_N; }
    int leadingDim() { return _T ? _ncols:_nrows; }
    int followingDim() { return _T ? _nrows:_ncols; }
    T* host_data() { return host_data; }
    T* dev_data() { return dev_data; }
    bool on_device() { return _on_device; }

    void host2dev() {
        CUBLAS_CALL(cublasSetMatrix(_nrows, _ncols, sizeof(T), _host_data, _nrows, _dev_data, _nrows)); 
    }

    void dev2host() {
        CUBLAS_CALL(cublasGetMatrix(_nrows, _ncols, sizeof(T), _dev_data, _nrows, _host_data, _nrows));
    }

    void update(DMatrix& A, DMatrix& B, const T alpha, const T beta) {
        if (_on_device) {
            cublasOperation_t transa = A.T() ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t transb = B.T() ? CUBLAS_OP_T : CUBLAS_OP_N;
            CUBLAS_CALL(cublasXgemm(_handle, transa, transb, A.nrows(), B.ncols(), A.ncols(), 
                                &alpha, A.dev_data(), A.leadingDim(), B.dev_data(), B.leadingDim()
                                &beta, _dev_data, leadingDim()));
        }else{
            exit(-1);
        }
    }
};


#endif //DMATRIX_CUH
