#ifndef DOPERATORS_CUH
#define DOPERATORS_CUH

#include <common.cuh>
#include <math_functions.h>

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
