#ifndef DNEURON_CUH
#define DNEURON_CUH

#include <common.cuh>
#include <DMatrix.cuh>

template<class T>
class DNeuron {
    class ForwardOp {
    public:
        __host__ __device__ inline T operator() (T act, T drv) {
            return drv;
        }
    }
    class BackwardOp {
    public:
        __host__ __device__ inline T operator() (T delta, T drv, T act) {
            return delta*1.0;
        }
    }
    class DeltaOp{
    public:
        __host__ __device__ inline T operator() (T x, T y, T z) {
            return y - z;
        }
    };


public:
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, _act.nelem() - _act.ld()); 
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act);
    }
    virtual T initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        delta->applyTenary(DeltaOp(), act, y);
        return delta.norm2(delta->nelem() - delta->ld())/delta->ld();
    }
}

template<class T>
class DReLUNeuron : DNeuron<T> {
    class ForwardOp {
    public:
        __host__ __device__ T operator() (T act, T drv) {
            return drv*(drv > (T)0.0);
        }
    }
    class BackwardOp {
    public:
        __host__ __device__ T operator() (T delta, T drv, T act) {
            return delta*(drv > (T)0.0);
        }
    }

public:
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        act->applyBinary(ForwardOp(), drv, _act.nelem() - _act.ld()); 
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        delta->applyTenary(BackwardOp(), drv, act);
    }
}

template<class T>
class DSoftmaxNeuron : DNeuron<T> {
public:
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
    }
    virtual void bprop(DMatrix<T>* delta, DMatrix<T>* drv, DMatrix<T>* act) {
        return;
    }
    virtual T initDelta(DMatrix<T> *delta, DMatrix<T> *act, DMatrix<T> *y) {
        delta->applyTenary(DeltaOp(), act, y);
        return delta.norm2(delta->nelem() - delta->ld())/delta->ld();
    }
}



#define //DNEURON_CUH
