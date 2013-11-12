#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <common.cuh>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <cuPrintf.cu>


template<class T, class Op, bool even_m, bool even_n, bool y_trans>
__global__ void kApplyBinaryOp(Op op, T* x, const T* y, int m, int n, int ldx, int ldy) {
    if (y_trans) 
}


template<class T, class Op, bool isMulti>
__global__ void kApplyTenaryOp(Op op, T* dest, const T* x, const T* y, int nelem) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    if (isMulti || i < nelem) {
        dest[i] = op(dest[i], x[i], y[i]);
    }
}

__global__ void kSetupCurand(curandState *state, int nelem, unsigned int seed);

template<class T, bool isMulti>
__global__ void kDropout(T *dest, curandState *state, int nelem, float rate) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (isMulti || i < nelem) {
        curandState localState = state[i];
        dest[i] *= curand_uniform(&localState) > rate;
        state[i] = localState;
    }
}

template<class T, int num_thrd>
__global__ void kSoftmaxAct(T *act, T *drv, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y*ld + i;
    const int blockSize = ld*num_thrd;

    int n = ld*fd;
    if (i < ld) {
        T mySum = 0;
        while (j < n) {
            mySum += exp(drv[j]);
            j += blockSize;
        }
        j = threadIdx.y;
        i = j*WARP_SIZE + threadIdx.x;
        smem[i] = mySum;
        __syncthreads();
        if (num_thrd >= 32) 
            if (j < 16)
                smem[i] = mySum = mySum + smem[i+16*WARP_SIZE];
        __syncthreads();
        if (num_thrd >= 16)
            if (j < 8)
                smem[i] = mySum = mySum + smem[i+8*WARP_SIZE];
        __syncthreads();
        if (num_thrd >= 8)
            if (j < 4)
                smem[i] = mySum = mySum + smem[i+4*WARP_SIZE];
        __syncthreads();
        if (num_thrd >= 4)
            if (j < 2)
                smem[i] = mySum = mySum + smem[i+2*WARP_SIZE];
        __syncthreads();
        mySum = smem[threadIdx.x] + smem[threadIdx.x+WARP_SIZE];
        i = blockIdx.x*WARP_SIZE + threadIdx.x;
        j = threadIdx.y*ld + i;
        while (j < n) {
            act[j] = exp(drv[j])/mySum;
            j += blockSize;
        }
       // cuPrintf("%dx%d: %f, %f\n", i, threadIdx.y, mySum, act[j-blockSize]); 
    }
}
#endif //KERNELS_CUH
