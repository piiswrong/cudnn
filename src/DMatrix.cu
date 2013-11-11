#include <common.cuh>
#include <kernels.cuh>
#include <DMatrix.cuh>

template<class T>
template<class Op>
void DMatrix<T>::applyBinary(Op op, DMatrix<T>* x, int nelem) {
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
    }else{
        T* x_data = x->host_data();
        for(int i = 0; i < _nelem; i++) {
            _host_data[i] = op(_host_data[i], x_data[i]);
        }
    }
}

template<class T>
template<class Op>
void DMatrix<T>::applyTenary(Op op, DMatrix<T>* x, DMatrix<T>* y, int nelem) {
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
    }else{
        T* x_data = x->host_data();
        T* y_data = y->host_data();
        for(int i = 0; i < _nelem; i++) {
            _dev_data[i] = op(_dev_data[i], x_data[i], y_data[i]);
        }
    }
}

