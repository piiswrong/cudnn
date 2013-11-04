#ifndef DMATRIX_CUH
#define DMATRIX_CUH

#include <common.cuh>
#include <memory.h>


template<class T> 
class DMatrix {
    cublasStatus_t _handle;
    int _ld, int _fd;
    int _nelem, _size;
    bool _T;
    T* _host_data;
    T* _dev_data;
    bool _on_device;
public:
    DMatrix(ld, fd, cublasStatus_t handle = 0) {
        _ld = ld;
        _fd = fd;
        _nelem = _ld*_fd;
        _size = _nelem*sizeof(T);
        _host_data = (T*)malloc(_size);
        if (handle) { 
            _on_device = true;
            _handle = handle;
            CUDA_CALL(cudaMalloc((void**)&_dev_data, _size));
        }
    }
    
    int nrows(bool t) { return T(t) ? fd:ld; }
    int ncols(bool t) { return T(t) ? ld:fd; }
    int nelem() { return _nelem; }
    bool T(bool t) { return _T^t; }
    void setT() { _T = !_T; }
    cublasOperation_t Tchar(bool t) { return _T^t ? CUBLAS_OP_T : CUBLAS_OP_N; }
    int ld() { return _ld; }
    int fd() { return _fd; }
    T* host_data() { return host_data; }
    T* dev_data() { return dev_data; }
    bool on_device() { return _on_device; }

    void host2dev() {
        CUBLAS_CALL(cublasSetMatrix(_ld, _fd, sizeof(T), _host_data, _ld, _dev_data, _ld)); 
    }

    void dev2host() {
        CUBLAS_CALL(cublasGetMatrix(_ld, _fd, sizeof(T), _dev_data, _ld, _host_data, _ld));
    }

    void add(DMatrix<T>* x, const T alpha, int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        cublasXaxpy(_handle, nelem, &alpha, x->dev_data(), 1, dev_data(), 1);
    }

    void update(DMatrix<T>* A, bool Ta, DMatrix<T>* B, bool Tb) {
        update(A, Ta, B, Tb, 1.0, 0.0);
    }

    void update(DMatrix<T>* A, bool Ta, DMatrix<T>* B, bool Tb, const T alpha, const T beta) {
        if (_on_device) {
            if (_T) {
               std::swap(A, B);
               std::swap(Ta, Tb);
               Ta != _T; Tb != _T;
            }
            CUBLAS_CALL(cublasXgemm(_handle, A->Tchar(Ta), B->Tchar(Tb), A->nrows(Ta), 
                                B->nrows(Tb), A->ncols(Ta), &alpha, 
                                A->dev_data(), A->ld(), B->dev_data(), B->ld()
                                &beta, _dev_data, ld()));
        }else{
            exit(-1);
        }
    }

    template<class Op>
    void applyBinary(Op op, DMatrix<T>* x, int nelem = 0) {
        if (nelem = 0) nelem = _nelem;
        if (_on_device) {
            if (nelem%BLOCK_SIZE == 0) {
                dim3 grid(nelem/BLOCK_SIZE);
                dim3 block(BLOCK_SIZE);
                kApplyBinaryOp<T,op><<<grid, block>>>(_dev_data, x->dev_data(), nelem, op);
                CUDA_CALL(cudaPeekAtLastError());
            }else {
                dim3 grid(nelem/BLOCK_SIZE + 1);
                dim3 block(BLOCK_SIZE);
                kApplyBinaryOpOdd<T,op><<<grid, block>>>(_dev_data, x->dev_data(), nelem, op);
                CUDA_CALL(cudaPeekAtLastError());
            }
        }else{
            T* x_data = x->host_data();
            for(int i = 0; i < _nelem; i++) {
                _host_data[i] = op(x_data[i]);
            }
        }
    }

    template<class Op>
    void applyTenary(Op op, DMatrix<T>* x, DMatrix<T>* y, int nelem = 0) {
        if (nelem = 0) nelem = _nelem;
        if (_on_device) {
            if (nelem%BLOCK_SIZE == 0) {
                dim3 grid(nelem/BLOCK_SIZE);
                dim3 block(BLOCK_SIZE);
                kApplyTenaryOp<T,op><<<grid, block>>>(_dev_data, x->dev_data(), y->dev_data(), nelem, op);
                CUDA_CALL(cudaPeekAtLastError());
            }else {
                dim3 grid(nelem/BLOCK_SIZE + 1);
                dim3 block(BLOCK_SIZE);
                kApplyTenaryOpOdd<T,op><<<grid, block>>>(_dev_data, dest->dest_data(), y->dev_data(), nelem, op);
                CUDA_CALL(cudaPeekAtLastError());
            }
        }else{
            T* x_data = x->host_data();
            T* y_data = y->host_data();
            for(int i = 0; i < _nelem; i++) {
                _dev_data[i] = op(_dev_data[i], x_data[i], y_data[i]);
            }
        }
    }
};


#endif //DMATRIX_CUH
