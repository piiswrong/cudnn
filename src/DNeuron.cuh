#ifndef DNEURON_CUH
#define DNEURON_CUH

#include <common.cuh>
#include <math_functions.h>
#include <DMatrix.cuh>
#include <DOperators.cuh>
#include <kernels.cuh>

template<class T>
class DNeuron {
protected:
    bool _on_device;
    cublasHandle_t _handle;
    T _loss;
public:
    class ForwardOp {
    public:
        __host__ __device__ inline T operator() (T act, T drv) {
            return drv;
        }
    };
    class BackwardOp {
    public:
        __host__ __device__ inline T operator() (T delta, T drv, T act) {
            return delta*1.0;
        }
    };
    class DeltaOp{
    public:
        __host__ __device__ inline T operator() (T x, T y, T z) {
            return y - z;
        }
    };

    DNeuron(cublasHandle_t handle) {
        _handle = handle;
        if (_handle != 0)
            _on_device = true;
        else
            _on_device = false;
    }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }
    virtual void initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        delta->applyTenary(DeltaOp(), act, y, delta->nrows(), delta->ncols() - 1);
    }
    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        _loss = delta->norm2(delta->nelem() - delta->ld());
    }
    virtual T getLoss() {
        return _loss;
    }
};

template<class T>
class DReLUNeuron : public DNeuron<T> {
public:
    class ForwardOp {
    public:
        __host__ __device__ T operator() (T act, T drv) {
            return drv*(drv > (T)0.0);
        }
    };
    class BackwardOp {
    public:
        __host__ __device__ T operator() (T delta, T drv, T act) {
            return delta*(drv > (T)0.0);
        }
    };

    DReLUNeuron(cublasHandle_t handle) : DNeuron<T>(handle) {}
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, act->nrows(), act->ncols() - 1);
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act, delta->nrows(), delta->ncols());
    }
};

template<class T>
class DSoftmaxNeuron : public DNeuron<T> {
protected:
    cudaStream_t _stream;
    DMatrix<int> *res;
    DMatrix<T> *_y;
public:
    DSoftmaxNeuron(int batch_size, cublasHandle_t handle) : DNeuron<T>(handle) {
        if (DNeuron<T>::_on_device) {
            CUDA_CALL(cudaStreamCreate(&_stream));
        }
        res = new DMatrix<int>(batch_size, 1, handle);
    }
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        if (DNeuron<T>::_on_device) {
            dim3 grid((act->ld()-1)/WARP_SIZE+1, 1, 1);
            dim3 block(WARP_SIZE, 32, 1);
            kSoftmaxAct<T,32><<<grid, block>>>(act->dev_data(), drv->dev_data(), res->dev_data(), act->ld(), act->fd()-1);
#ifndef NDEBUG
            act->dev2host();
#endif
        }else {
            exit(-1);
        }
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        return;
    }
    virtual void computeLoss(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        res->dev2hostAsync(_stream);
        _y = y;
        //act->applyTenary(OpWeighted<T>(), act, y, act->nrows(), act->ncols());
        //DNeuron<T>::_loss = act->norm1(act->nelem() - act->ld());
    }
    virtual T getLoss() {
        cudaStreamSynchronize(_stream);
        T loss = 0;
        for (int i = 0; i < _y->nrows(); i++) loss += _y->getElem(i, res->getElem(i, 0));
        return loss;
    }
};



#endif //DNEURON_CUH
