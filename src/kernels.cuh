#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <common.cuh>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <cuPrintf.cu>


template<class T, class Op, bool even_m, bool even_n, bool y_trans>
__global__ void kApplyBinaryOp(Op op, T* x, const T* y, int m, int n, int ldx, int ldy) {
    int i, j;
    if (y_trans) {
        __shared__ T tile[TILE_DIM][TILE_DIM + 1];
        i = blockIdx.y * TILE_DIM + threadIdx.x;
        j = blockIdx.x * TILE_DIM + threadIdx.y;
        if (even_n || i < n) 
            for (int k = 0; (even_m || j < m) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                tile[threadIdx.y + k][threadIdx.x] = y[i+j*ldy];
        i = blockIdx.x * TILE_DIM + threadIdx.x;
        j = blockIdx.y * TILE_DIM + threadIdx.y;
        __syncthreads();
        if (even_m || i < m) 
            for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                x[i+j*ldx] = op(x[i+j*ldx], tile[threadIdx.x][threadIdx.y + k]);
    }else {
        i = blockIdx.x * TILE_DIM + threadIdx.x;
        j = blockIdx.y * TILE_DIM + threadIdx.y;
        if (even_m || i < m) 
            for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                x[i+j*ldx] = op(x[i+j*ldx], y[i+j*ldy]);
    }
}

//TODO: load y,z togather if possible.
template<class T, class Op, bool even_m, bool even_n, bool y_trans, bool z_trans>
__global__ void kApplyTenaryOp(Op op, T* x, const T* y, const T* z, int m, int n, int ldx, int ldy, int ldz) {
    int i, j;
    __shared__ T tile_y[TILE_DIM][TILE_DIM+1];
    __shared__ T tile_z[TILE_DIM][TILE_DIM+1];
    if (z_trans) {
        i = blockIdx.y * TILE_DIM + threadIdx.x;
        j = blockIdx.x * TILE_DIM + threadIdx.y;
        if (even_n || i < n) 
            for (int k = 0; (even_m || j < m) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                tile_z[threadIdx.y + k][threadIdx.x] = z[i+j*ldz];
    }else {
        i = blockIdx.x * TILE_DIM + threadIdx.x;
        j = blockIdx.y * TILE_DIM + threadIdx.y;
        if (even_m || i < m) 
            for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                tile_z[threadIdx.x][threadIdx.y + k] = z[i+j*ldz];
    }
    
    if (y_trans) {
        i = blockIdx.y * TILE_DIM + threadIdx.x;
        j = blockIdx.x * TILE_DIM + threadIdx.y;
        if (even_n || i < n) 
            for (int k = 0; (even_m || j < m) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                tile_y[threadIdx.y + k][threadIdx.x] = y[i+j*ldy];
    }else {
        i = blockIdx.x * TILE_DIM + threadIdx.x;
        j = blockIdx.y * TILE_DIM + threadIdx.y;
        if (even_m || i < m) 
            for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                tile_y[threadIdx.x][threadIdx.y + k] = y[i+j*ldy];
    }

    i = blockIdx.x * TILE_DIM + threadIdx.x;
    j = blockIdx.y * TILE_DIM + threadIdx.y;
    if (z_trans || y_trans) __syncthreads();
    if (even_m || i < m) 
        for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
            x[i+j*ldx] = op(x[i+j*ldx], tile_y[threadIdx.x][threadIdx.y + k], tile_z[threadIdx.x][threadIdx.y + k]);
}

__global__ void kSetupCurand(curandState *state, int nelem, unsigned int seed);

template<class T, bool even_m, bool even_n>
__global__ void kDropout(T *dest, curandState *state, float rate, int m, int n, int ld) { 
    int i = blockIdx.x * TILE_DIM + threadIdx.x;
    int j = blockIdx.y * TILE_DIM + threadIdx.y;
    if (even_m || i < m) {
        curandState localState = state[i+j*m];
        for (int k = 0; (even_n || j + k < n) && k < TILE_DIM; k += BLOCK_ROWS) {
            dest[i+(j+k)*ld] *= curand_uniform(&localState) > rate;
        }
        state[i+j*m] = localState;
    }
}

template<class T> 
void hDropout(T *x, curandState *state, float rate, bool trans, int m, int n, int ld) {
    if (trans) std::swap(m, n);
    bool even_m = !(m%TILE_DIM), even_n = !(n%TILE_DIM);
    dim3 grid(m/TILE_DIM+!even_m, n/TILE_DIM+!even_n, 1);
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    switch((even_m<<1)|even_n) {
    case 0: kDropout<T, false, false><<<grid, block>>>(x, state, rate, m, n, ld);break;
    case 1: kDropout<T, false, true><<<grid, block>>>(x, state, rate, m, n, ld);break;
    case 2: kDropout<T, true, false><<<grid, block>>>(x, state, rate, m, n, ld);break;
    case 3: kDropout<T, true, true><<<grid, block>>>(x, state, rate, m, n, ld);break;
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
        T myMax = -1e37;
        while (j < n) {
            if (myMax < drv[j]) myMax = drv[j];
            j += blockSize;
        }
        j = threadIdx.y;
        i = j*WARP_SIZE + threadIdx.x;
        smem[i] = myMax;
        __syncthreads();
        if (num_thrd >= 32) {
            if (j < 16)
                if (myMax < smem[i+16*WARP_SIZE])
                    smem[i] = myMax = smem[i+16*WARP_SIZE];
             __syncthreads();
        }
        if (num_thrd >= 16) {
            if (j < 8)
                if (myMax < smem[i+8*WARP_SIZE])
                    smem[i] = myMax = smem[i+8*WARP_SIZE];
            __syncthreads();
        }
        if (num_thrd >= 8) {
            if (j < 4)
                if (myMax < smem[i+4*WARP_SIZE])
                    smem[i] = myMax = smem[i+4*WARP_SIZE];
            __syncthreads();
        }
        if (num_thrd >= 4) {
            if (j < 2)
                if (myMax < smem[i+2*WARP_SIZE])
                    smem[i] = myMax = smem[i+2*WARP_SIZE];
            __syncthreads();
        }
        myMax = smem[threadIdx.x] > smem[threadIdx.x+WARP_SIZE] ? smem[threadIdx.x] : smem[threadIdx.x+WARP_SIZE];

        i = blockIdx.x*WARP_SIZE + threadIdx.x;
        j = threadIdx.y*ld + i;
        T mySum = 0;
        while (j < n) {
            mySum += exp(drv[j]-myMax);
            j += blockSize;
        }
        j = threadIdx.y;
        i = j*WARP_SIZE + threadIdx.x;
        __syncthreads();
        smem[i] = mySum;
        __syncthreads();
        if (num_thrd >= 32) {
            if (j < 16)
                smem[i] = mySum = mySum + smem[i+16*WARP_SIZE];
            __syncthreads();
        }
        if (num_thrd >= 16) {
            if (j < 8)
                smem[i] = mySum = mySum + smem[i+8*WARP_SIZE];
            __syncthreads();
        }
        if (num_thrd >= 8) {
            if (j < 4)
                smem[i] = mySum = mySum + smem[i+4*WARP_SIZE];
            __syncthreads();
        }
        if (num_thrd >= 4) {
            if (j < 2)
                smem[i] = mySum = mySum + smem[i+2*WARP_SIZE];
            __syncthreads();
        }
        mySum = smem[threadIdx.x] + smem[threadIdx.x+WARP_SIZE];
        i = blockIdx.x*WARP_SIZE + threadIdx.x;
        j = threadIdx.y*ld + i;
        while (j < n) {
            act[j] = exp(drv[j]-myMax)/mySum;
            j += blockSize;
        }
       // cuPrintf("%dx%d: %f, %f\n", i, threadIdx.y, mySum, act[j-blockSize]); 
    }
}
#endif //KERNELS_CUH
