#ifndef DOPERATORS_CUH
#define DOPERATORS_CUH

#include <common.cuh>
#include <math_functions.h>

template<class T>
class OpScale{
    const T _scale;
public:
    OpScale(const T scale) : _scale(scale) {}
    __host__ __device__ T operator() (T x, T y) {
        return y*_scale;
    }
};


template<class T>
class OpWeightedLog{
public:
    __host__ __device__ T operator() (T x, T y, T z) {
        return log(y)*z;
    }
};

template<class T>
class OpWeighted{
public:
    __host__ __device__ T operator() (T x, T y, T z) {
        return y*z;
    }
};


template<class T>
class OpLog{
public:
    __host__ __device__ T operator() (T x, T y) {
        return log(y);
    }
};


template<class T>
class OpExp{
public:
    __host__ __device__ T operator() (T x, T y) {
        return exp(y);
    }
};


template<class T>
class OpSub{
public:
    __host__ __device__ T operator() (T x, T y, T z) {
        return y - z;
    }
};


#endif //DOPERATORS_CUH
