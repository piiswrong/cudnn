#ifndef DMATRIX_CUH
#define DMATRIX_CUH

#include <common.cuh>
#include <kernels.cuh>
#include <DRand.h>

template<class T> 
class DMatrix {
    cublasHandle_t _handle;
    bool _is_view;

    int _ld, _fd, _nelem, _size;
    bool _T;
    T* _host_data;
    T* _dev_data;
    bool _on_device;
	
public:
    enum Init {
        None = 0,
        Zero = 1,
        Uniform = 2,
        Normal = 4,
        ColSparse = 1+8,
        RowSparse = 1 + 16,
        Weight = 1+32,
        One = 64
    };
    DMatrix(int ld, int fd, cublasHandle_t handle = 0) {
        _is_view = false;
        _ld = ld;
        _fd = fd;
        _T = false;
        _nelem = _ld*_fd;
        _size = _nelem*sizeof(T);
        _host_data = (T*)malloc(_size);
        _handle = handle;
        if (_handle) { 
            _on_device = true;
            CUDA_CALL(cudaMalloc((void**)&_dev_data, _size));
        }
    }

    DMatrix(DMatrix<T>* x, int offset, int length) {
        _is_view = true;
        _ld = x->_ld;
        _fd = length;
        _T = x->_T;
        _nelem = _ld*length;
        _size = _nelem*sizeof(T);
        _host_data = x->_host_data + _ld*offset;
        _handle = x->handle();
        if (_handle) {
            _on_device = true;
            _dev_data = x->_dev_data + _ld*offset;
        }
    }

    ~DMatrix() {
        if (!_is_view) {
            free(_host_data);
            CUDA_CALL(cudaFree(_dev_data));
        }   
    }
    
    int nrows(bool t = false) { return getT(t) ? fd():ld(); }
    int ncols(bool t = false) { return getT(t) ? ld():fd(); }
    int nelem() { return _nelem; }
    int size() { return _size; }
    bool getT(bool t) { return _T^t; }
    void setT() { _T = !_T; }
    cublasOperation_t Tchar(bool t) { return _T^t ? CUBLAS_OP_T : CUBLAS_OP_N; }
    int ld() { return _ld; }
    int fd() { return _fd; }
    T* host_data() { return _host_data; }
    T* dev_data() { return _dev_data; }
    bool on_device() { return _on_device; }
    cublasHandle_t handle() { return _handle; }

	void init(int p, T a = 0.0, T b = 0.0) { 
		if (p&DMatrix<T>::Zero) memset(_host_data, 0, _size);
        if (p&DMatrix<T>::One) for (int i = 0; i < _nelem; i++) _host_data[i] = (T)1;
		if (p&DMatrix<T>::Uniform) {
            DRand dist(a, b);
			for (int i = 0; i < _nelem; i++) _host_data[i] = dist.uniform();
		}
		if (p&DMatrix<T>::Normal) {
            DRand dist(a, b);
			for (int i = 0; i < _nelem; i++) _host_data[i] = dist.normal();
		}
		if (p&DMatrix<T>::ColSparse) {
			for (int col = 0; col < _fd; col++) {
                int n1 = SPARSE_DEGREE, n2 = _ld - SPARSE_DEGREE;
                for (int row = 0; row < _ld; row++) {
                    int r = rand()%(n1+n2);
                    if (r<n1) {
                        n1--;
                    }else {
                        n2--;
                        _host_data[col*_ld + row] = 0.0;
                    }
                }
			}
		}
        if (p&DMatrix<T>::RowSparse) {
			for (int row = 0; row < _ld; row++) {
                int n1 = SPARSE_DEGREE, n2 = _fd - SPARSE_DEGREE;
                for (int col = 0; col < _fd; col++) {
                    int r = rand()%(n1+n2);
                    if (r<n1) {
                        n1--;
                    }else {
                        n2--;
                        _host_data[col*_ld + row] = 0.9;
                    }
                }
			}
		}
        if (p&DMatrix<T>::Weight) {
            for (int i = _nelem - _ld; i < _nelem; i++) _host_data[i] = 0.0;
            _host_data[_nelem-1] = 1.0;
        }
        if (_on_device) host2dev();
	}
    
    void host2dev() {
        CUBLAS_CALL(cublasSetVector(_ld*_fd, sizeof(T), _host_data, 1, _dev_data, 1)); 
    }

    void dev2host() {
        CUBLAS_CALL(cublasGetVector(_ld*_fd, sizeof(T), _dev_data, 1, _host_data, 1));
    }

    void host2devAsync(cudaStream_t stream) {
        CUBLAS_CALL(cublasSetVectorAsync(_ld*_fd, sizeof(T), _host_data, 1, _dev_data, 1, stream)); 
    }

    void dev2hostAsync(cudaStream_t stream) {
        CUBLAS_CALL(cublasGetVectorAsync(_ld*_fd, sizeof(T), _dev_data, 1, _host_data, 1, stream));
    }

    T norm1(int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        T res = 0.0;
        if (_on_device) {
            CUBLAS_CALL(cublasXasum(_handle, nelem, _dev_data, 1, &res));
        }else {
            for (int i = 0; i < nelem; i++) res += abs(_host_data[i]);
        }
        return res;
    }


    T norm2(int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        T res = 0.0;
        if (_on_device) {
            CUBLAS_CALL(cublasXnrm2(_handle, nelem, _dev_data, 1, &res));
        }else {
            for (int i = 0; i < nelem; i++) res += _host_data[i]*_host_data[i];
        }
        return res;
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
               Ta = !Ta; Tb = !Tb;
            }
            CUBLAS_CALL(cublasXgemm(_handle, A->Tchar(Ta), B->Tchar(Tb), A->nrows(Ta), 
                                B->nrows(Tb), A->ncols(Ta), &alpha, 
                                A->dev_data(), A->ld(), B->dev_data(), B->ld(),
                                &beta, _dev_data, ld()));
#ifdef DEBUG_BUILD
            dev2host();
#endif
        }else{
            exit(-1);
        }
    }

    template<class Op>
    void applyBinary(Op op, DMatrix<T>* x, int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        if (_on_device) {
            if (nelem%BLOCK_SIZE == 0) {
                dim3 grid(nelem/BLOCK_SIZE, 1, 1);
                dim3 block(BLOCK_SIZE, 1, 1);
                kApplyBinaryOp<T, Op, true><<<grid, block>>>(op, _dev_data, x->dev_data(), nelem);
                CUDA_CALL(cudaPeekAtLastError());
            }else {
                dim3 grid(nelem/BLOCK_SIZE + 1, 1, 1);
                dim3 block(BLOCK_SIZE, 1, 1);
                kApplyBinaryOp<T, Op, false><<<grid, block>>>(op, _dev_data, x->dev_data(), nelem);
                CUDA_CALL(cudaPeekAtLastError());
            }
#ifdef DEBUG_BUILD
            dev2host();
#endif
        }else{
            T* x_data = x->host_data();
            for(int i = 0; i < _nelem; i++) {
                _host_data[i] = op(_host_data[i], x_data[i]);
            }
        }
    }

    template<class Op>
    void applyTenary(Op op, DMatrix<T>* x, DMatrix<T>* y, int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        if (_on_device) {
            if (nelem%BLOCK_SIZE == 0) {
                dim3 grid(nelem/BLOCK_SIZE, 1, 1);
                dim3 block(BLOCK_SIZE, 1, 1);
                kApplyTenaryOp<T, Op, true><<<grid, block>>>(op, _dev_data, x->dev_data(), y->dev_data(), nelem);
                CUDA_CALL(cudaPeekAtLastError());
            }else {
                dim3 grid(nelem/BLOCK_SIZE + 1, 1 ,1);
                dim3 block(BLOCK_SIZE, 1, 1);
                kApplyTenaryOp<T, Op, false><<<grid, block>>>(op, _dev_data, x->dev_data(), y->dev_data(), nelem);
                CUDA_CALL(cudaPeekAtLastError());
            }
#ifdef DEBUG_BUILD
            dev2host();
#endif
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
