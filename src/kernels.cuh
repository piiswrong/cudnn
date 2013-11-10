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

template<class T, int num_thrd>
__global__ void kSoftmaxAct(T *act, T *drv, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    int i = blockDim.x*WARP_SIZE + threadIdx.x;
    const int blockSize = WARP_SIZE*num_thrd;

    if (i < ld) {
        T mySum = 0;
        int n = ld*fd;
        while (i < n) {
            mySum += exp(drv[i]);
            i += blockSize;
        }
        int j = threadIdx.y;
        i = j*WARP_SIZE + threadIdx.x;
        smem[i] = mySum;
        __syncthreads();
        if (num_thrd >= 32) 
            if (j < 16)
                smem[i] = mySum = mySum + smem[i+16];
        __syncthreads();
        if (num_thrd >= 16)
            if (j < 8)
                smem[i] = mySum = mySum + smem[i+8];
        __syncthreads();
        if (num_thrd >= 8)
            if (j < 4)
                smem[i] = mySum = mySum + smem[i+4];
        __syncthreads();
        if (num_thrd >= 4)
            if (j < 2)
                smem[i] = mySum = mySum + smem[i+2];
        __syncthreads();
        mySum = smem[threadIdx.x] + smem[threadIdx.x+WARP_SIZE];
        i = blockDim.x*WARP_SIZE + threadIdx.x;
        while (j < fd) {
            act[i] = exp(drv[i])/mySum;
            j += blockSize;
        }
    }
}
#endif //KERNELS_CUH
