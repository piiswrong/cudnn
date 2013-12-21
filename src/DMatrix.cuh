#ifndef DMATRIX_CUH
#define DMATRIX_CUH


#include <common.cuh>
#include <kernels.cuh>


template<class T> 
class DMatrix {
    cublasHandle_t _handle;
    bool _is_view;

    int _ld, _fd, _nelem, _size;
    bool _T;
    T* _host_data;
    T* _dev_data;
    bool _on_device;

#ifdef USE_MPI
    MPI_Win _win;
#endif
	
    enum _Init {
        _None = 0,
        _Zero = 1,
        _Uniform = 2,
        _Normal = 4,
        _ColSparse = 8,
        _RowSparse = 16,
        _Weight = 32,
        _One = 64
    };

public:
    enum Init {
        None = 0,
        Zero = 1,
        Uniform = 2,
        //Normal = 4,
        ColSparse = 1+8,
        RowSparse = 1 + 16,
        Weight = 1+32,
        One = 64
    };
    DMatrix(int ld, int fd, cublasHandle_t handle = 0) {
        _on_device = false;
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
        _on_device = false;
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
            if (_on_device) {
                CUDA_CALL(cudaFree(_dev_data));
            }
        }   
    }

    void samplePrint() {
        for (int i = 0; i < 10 && i < nrows(); i++) {
            for (int j = 0; j < 10 && j < ncols(); j++) {
                printf("%+0.3f ", getElem(i,j));
            }
            printf("\n");
        }
        printf("\n");
    }
    
    int nrows(bool t = false) { return getT(t) ? fd():ld(); }
    int ncols(bool t = false) { return getT(t) ? ld():fd(); }
    int nelem() { return _nelem; }
    int size() { return _size; }
    bool getT(bool t = false) { return _T^t; }
    void setT() { _T = !_T; }
#ifndef DISABLE_GPU
    cublasOperation_t Tchar(bool t) { return getT(t) ? CUBLAS_OP_T : CUBLAS_OP_N; }
#endif
    CBLAS_TRANSPOSE hTchar(bool t) { return getT(t) ? CblasTrans : CblasNoTrans; }
    int ld() { return _ld; }
    int fd() { return _fd; }
    T* host_data() { return _host_data; }
    T* dev_data() { return _dev_data; }
    bool on_device() { return _on_device; }
    cublasHandle_t handle() { return _handle; }
    T& getElem(int i, int j) {
        if (getT()) std::swap(i, j);
        return _host_data[i+j*ld()];
    }

	void init(int p, T a = 0.0, T b = 0.0) { 
		if (p&_Zero) memset(_host_data, 0, _size);
        if (p&_One) for (int i = 0; i < _nelem; i++) _host_data[i] = (T)1;
		if (p&_Uniform) {
			for (int i = 0; i < _nelem; i++) 
                _host_data[i] = ((T)rand())/RAND_MAX*(b-a) + a;
		}
		/*if (p&_Normal) {
            DRand dist(a, b);
			for (int i = 0; i < _nelem; i++) 
                _host_data[i] = dist.normal();
		}*/
		if (p&_ColSparse) {
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
        if (p&_RowSparse) {
			for (int row = 0; row < _ld; row++) {
                int n1 = SPARSE_DEGREE, n2 = _fd - SPARSE_DEGREE;
                for (int col = 0; col < _fd; col++) {
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
        if (p&_Weight) {
            for (int i = _nelem - _ld; i < _nelem; i++) _host_data[i] = 0.0;
            _host_data[_nelem-1] = 1.0;
        }
        host2dev();
	}
    
    void host2dev() {
        if (_on_device)
            CUBLAS_CALL(cublasSetVector(_ld*_fd, sizeof(T), _host_data, 1, _dev_data, 1)); 
    }

    void dev2host() {
        if (_on_device)
            CUBLAS_CALL(cublasGetVector(_ld*_fd, sizeof(T), _dev_data, 1, _host_data, 1));
    }

    void host2devAsync(cudaStream_t stream) {
        if (_on_device) 
            CUBLAS_CALL(cublasSetVectorAsync(_ld*_fd, sizeof(T), _host_data, 1, _dev_data, 1, stream)); 
            //CUDA_CALL(cudaMemcpyAsync(_dev_data, _host_data, _size, cudaMemcpyHostToDevice, stream));
    }

    void dev2hostAsync(cudaStream_t stream) {
        if (_on_device) 
            CUBLAS_CALL(cublasGetVectorAsync(_ld*_fd, sizeof(T), _dev_data, 1, _host_data, 1, stream));
    }

    void CopyFrom(DMatrix<T> *src) {
        if (_on_device) 
            CUDA_CALL(cudaMemcpy(_dev_data, src->dev_data(), _size, cudaMemcpyDeviceToDevice));
        else
            memcpy(_host_data, src->host_data(), _size);
    }

#ifdef USE_MPI
    MPI_Datatype mpiDatatype() {
        printf("Datatype not supported by MPI\n");
        exit(-1);
    }

    void mpiAllReduce(DMatrix<T> *dest, MPI_Op op) {
        dev2host();
        MPI_Allreduce(MPI_IN_PLACE, dest->host_data(), nelem(), mpiDatatype(), op, MPI_COMM_WORLD);
        dest->host2dev();
    }

    void mpiSend(int dest, int tag) {
        dev2host();
        MPI_Send(host_data(), nelem(), mpiDatatype(), dest, tag, MPI_COMM_WORLD);
    }

    void mpiRecv(int src, int tag) {
        MPI_Status status;
        MPI_Recv(host_data(), nelem(), mpiDatatype(), src, tag, MPI_COMM_WORLD, &status);
        host2dev();
    }    

    void mpiCreateWin() {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Win_create(host_data(), nelem(), sizeof(T), info, MPI_COMM_WORLD, &_win);
        MPI_Info_free(&info);
    }

    void mpiGet(int target) {
	MPI_Win_lock(MPI_LOCK_SHARED, target, 0, _win);
        MPI_Get(host_data(), nelem(), mpiDatatype(), target, 0, nelem(), mpiDatatype(), _win);
	MPI_Win_unlock(target, _win);
        host2dev();
    }
#endif

    T norm1(int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        T res = 0.0;
        if (_on_device) {
            CUBLAS_CALL(cublasXasum(_handle, nelem, _dev_data, 1, &res));
        }else {
            res = cblas_Xasum(nelem, host_data(), 1);
        }
        return res;
    }


    T norm2(int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        T res = 0.0;
        if (_on_device) {
            CUBLAS_CALL(cublasXnrm2(_handle, nelem, _dev_data, 1, &res));
        }else {
            res = cblas_Xnrm2(nelem, host_data(), 1);
        }
        return res;
    }

    void add(DMatrix<T>* x, const T alpha, int nelem = 0) {
        if (nelem == 0) nelem = _nelem;
        if (_on_device) {
            CUBLAS_CALL(cublasXaxpy(_handle, nelem, &alpha, x->dev_data(), 1, dev_data(), 1));
        }else {
            cblas_Xaxpy(nelem, alpha, x->host_data(), 1, host_data(), 1);
        }
#ifndef NDEBUG
        dev2host();
#endif
    }

    void update(DMatrix<T>* A, bool Ta, DMatrix<T>* B, bool Tb) {
        update(A, Ta, B, Tb, 1.0, 0.0);
    }

    void update(DMatrix<T>* A, bool Ta, DMatrix<T>* B, bool Tb, const T alpha, const T beta) {
        if (_T) {
            std::swap(A, B);
            std::swap(Ta, Tb);
            Ta = !Ta; Tb = !Tb;
        }
        if (_on_device) {
            CUBLAS_CALL(cublasXgemm(_handle, A->Tchar(Ta), B->Tchar(Tb), A->nrows(Ta), 
                                B->ncols(Tb), A->ncols(Ta), &alpha, 
                                A->dev_data(), A->ld(), B->dev_data(), B->ld(),
                                &beta, _dev_data, ld()));
#ifndef NDEBUG
            dev2host();
#endif
        }else{
            cblas_Xgemm(CblasColMajor, A->hTchar(Ta), B->hTchar(Tb), A->nrows(Ta),
                        B->ncols(Tb), A->ncols(Ta), alpha,
                        A->host_data(), A->ld(), B->host_data(), B->ld(),
                        beta, host_data(), ld());
        }
    }

    template<class Op>
    void applyBinary(Op op, DMatrix<T>* y, int m, int n) {
#ifndef DISABLE_GPU
        if (_on_device) {
            if (getT()) std::swap(m, n);
            bool y_trans = y->getT(getT());
            int ldx = ld(), ldy = y->ld();
            bool even_m = !(m%TILE_DIM), even_n = !(n%TILE_DIM);
            dim3 grid(m/TILE_DIM+!even_m, n/TILE_DIM+!even_n, 1);
            dim3 block(TILE_DIM, BLOCK_ROWS, 1);
            int word = (even_m<<2)|(even_n<<1)|y_trans;
            switch(word) {
            case 0: kApplyBinaryOp<T, Op, false, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 1: kApplyBinaryOp<T, Op, false, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 2: kApplyBinaryOp<T, Op, false, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 3: kApplyBinaryOp<T, Op, false, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 4: kApplyBinaryOp<T, Op, true, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 5: kApplyBinaryOp<T, Op, true, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 6: kApplyBinaryOp<T, Op, true, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            case 7: kApplyBinaryOp<T, Op, true, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), m, n, ldx, ldy);CUDA_KERNEL_CHECK();break;;
            }
#ifndef NDEBUG
            dev2host();
#endif
        }else
#endif
        {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    getElem(i,j) = op(getElem(i,j), y->getElem(i,j));               
                }
            }
        }
    }

    template<class Op>
    void applyTenary(Op op, DMatrix<T>* y, DMatrix<T>* z, int m, int n) {
#ifndef DISABLE_GPU
        if (_on_device) {
            if (getT()) std::swap(m, n);
            bool y_trans = y->getT(_T), z_trans = z->getT(_T);
            int ldx = ld(), ldy = y->ld(), ldz = z->ld();
            bool even_m = !(m%TILE_DIM), even_n = !(n%TILE_DIM);
            dim3 grid(m/TILE_DIM+!even_m, n/TILE_DIM+!even_n, 1);
            dim3 block(TILE_DIM, BLOCK_ROWS, 1);
            int word = (even_m<<3)|(even_n<<2)|(y_trans<<1)|(z_trans);
            switch(word) {
            case 0: kApplyTenaryOp<T, Op, false, false, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 1: kApplyTenaryOp<T, Op, false, false, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 2: kApplyTenaryOp<T, Op, false, false, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 3: kApplyTenaryOp<T, Op, false, false, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 4: kApplyTenaryOp<T, Op, false, true, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 5: kApplyTenaryOp<T, Op, false, true, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 6: kApplyTenaryOp<T, Op, false, true, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 7: kApplyTenaryOp<T, Op, false, true, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 8: kApplyTenaryOp<T, Op, true, false, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 9: kApplyTenaryOp<T, Op, true, false, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 10: kApplyTenaryOp<T, Op, true, false, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 11: kApplyTenaryOp<T, Op, true, false, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 12: kApplyTenaryOp<T, Op, true, true, false, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 13: kApplyTenaryOp<T, Op, true, true, false, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 14: kApplyTenaryOp<T, Op, true, true, true, false><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            case 15: kApplyTenaryOp<T, Op, true, true, true, true><<<grid, block>>>(op, dev_data(), y->dev_data(), z->dev_data(), m, n, ldx, ldy, ldz);CUDA_KERNEL_CHECK();break;;
            }
#ifndef NDEBUG
            dev2host();
#endif
        }else
#endif
        {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    getElem(i,j) = op(getElem(i,j), y->getElem(i,j), z->getElem(i,j));               
                }
            }
        }
    }
};

#ifdef USE_MPI
template<> MPI_Datatype DMatrix<float>::mpiDatatype() { return MPI_FLOAT; }
template<> MPI_Datatype DMatrix<double>::mpiDatatype() { return MPI_DOUBLE; }
template<> MPI_Datatype DMatrix<int>::mpiDatatype() { return MPI_INT; }
#endif

#ifndef DISABLE_GPU
template<class T> 
void hDropout(DMatrix<T> *x, DMatrix<T> *mask, curandState *state, float rate, bool trans, int m, int n, int ld) {
    if (trans) std::swap(m, n);
    bool even_m = !(m%TILE_DIM), even_n = !(n%TILE_DIM), save = (mask!=NULL);
    dim3 grid(m/TILE_DIM+!even_m, n/TILE_DIM+!even_n, 1);
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    switch((save<<2)|(even_m<<1)|even_n) {
    case 0: kDropout<T, false, false, false><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 1: kDropout<T, false, true, false><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 2: kDropout<T, true, false, false><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 3: kDropout<T, true, true, false><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 4: kDropout<T, false, false, true><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 5: kDropout<T, false, true, true><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 6: kDropout<T, true, false, true><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    case 7: kDropout<T, true, true, true><<<grid, block>>>(x->dev_data(), save?mask->dev_data():NULL, state, rate, m, n, ld);CUDA_KERNEL_CHECK();break;;
    }
}
#else
template<class T> 
void hDropout(DMatrix<T> *x, DMatrix<T> *mask, curandState *state, float rate, bool trans, int m, int n, int ld) {
    if (trans) std::swap(m,n);
    T* xdata = x->host_data();
    int thresh = rate*RAND_MAX;
    if (mask != NULL) {
        T* maskdata = mask->host_data();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                maskdata[i+m*j] = rand() > thresh;
                xdata[i+ld*j] *= maskdata[i+m*j];
            }
        }
    }else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                xdata[i+ld*j] *= rand() > thresh;
            }
        }
    }
}
#endif

#endif //DMATRIX_CUH
