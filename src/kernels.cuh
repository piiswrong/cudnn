#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <curand_kernel.h>

template<class T, class Op>
__global__ void kApplyBinaryOp(T* dest, const T* x, int nelem, Op op) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = op(dest[i], x[i]);
}


template<class T, class Op>
__global__ void kApplyBinaryOpOdd(T* dest, const T* x, int nelem, Op op) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        dest[i] = op(dest[i], x[i]);
    }
}

template<class T, class Op>
__global__ void kApplyTenaryOp(T* dest, const T* x, const T* y, int nelem, Op op) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = op(dest[i], x[i], y[i]);
}


template<class T, class Op>
__global__ void kApplyBinaryOpOdd(T* dest, const T* x, const T* y, int nelem, Op op) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        dest[i] = op(dest[i], x[i], y[i]);
    }
}

__global__ void kSetupCurand(curandState *state, int nelem, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) curand_init(seed, i, 0, &state[i]);
}

template<class T>
__global__ void kDropout(T *dest, curandState *state, int nelem, float rate) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        curandState localState = state[i];
        dest[i] *= curand_uniform(&localState) > rate;
        state[i] = localState;
    }
}

#endif //KERNELS_CUH
