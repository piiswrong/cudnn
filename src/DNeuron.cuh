#ifndef DNEURON_CUH
#define DNEURON_CUH

#include <common.cuh>
#include <DMatrix.cuh>

template<class T>
class DNeuron {
    class ForwardOp {
    public:
        __host__ __device__ T operator() (T act, T drv) {
            return drv;
        }
    }
    class BackwardOp {
    public:
        __host__ __device__ T operator() (T delta, T drv, T act) {
            return delta*1.0;
        }
    }

public:
    virtual void fprop(DMatrix<T>* act, DMatrix<T>* drv) {
        drv->apply(
    }
}



#define //DNEURON_CUH
