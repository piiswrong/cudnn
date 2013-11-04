#ifndef DOPERATORS_CUH
#define DOPERATORS_CUH

#include <common.cuh>

template<class T>
class OpSub{
public:
    __host__ __device__ T operator() (T x, T y, T z) {
        return y - z;
    }
};


#endif //DOPERATORS_CUH
