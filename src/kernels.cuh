#ifndef KERNELS_CUH
#define KERNELS_CUH
#ifndef DISABLE_GPU

#include <common.cuh>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


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

template<class T, bool even_m, bool even_n, bool save>
__global__ void kDropout(T *dest, T *mask, curandState *state, float rate, int m, int n, int ld) { 
    int i = blockIdx.x * TILE_DIM + threadIdx.x;
    int j = blockIdx.y * TILE_DIM + threadIdx.y;
    if (even_m || i < m) {
        curandState localState = state[i+j*m];
        for (int k = 0; (even_n || j + k < n) && k < TILE_DIM; k += BLOCK_ROWS) {
            if (save) {
                float t = curand_uniform(&localState) > rate;
                dest[i+(j+k)*ld] *= t;
                mask[i+(j+k)*ld] = t;
            }else {
                dest[i+(j+k)*ld] *= curand_uniform(&localState) > rate;
            }
        }
        state[i+j*m] = localState;
    }
}


template<class T, int num_thrd>
__global__ void kSoftmaxAct(T *act, T *drv, int *res, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    __shared__ int mark[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y*ld + i;
    const int blockSize = ld*num_thrd;

    int n = ld*fd;
    if (i < ld) {
        T myMax = -1e37;
        int myMark = 0;
        int k = threadIdx.y;
        while (j < n) {
            if (myMax < drv[j]) {
                myMax = drv[j];
                myMark = k;
            }
            k += num_thrd;
            j += blockSize;
        }
        j = threadIdx.y;
        i = j*WARP_SIZE + threadIdx.x;
        smem[i] = myMax;
        mark[i] = myMark;
        __syncthreads();
        if (num_thrd >= 32) {
            if (j < 16)
                if (myMax < smem[i+16*WARP_SIZE]) {
                    smem[i] = myMax = smem[i+16*WARP_SIZE];
                    mark[i] = myMark = mark[i+16*WARP_SIZE];
                }
             __syncthreads();
        }
        if (num_thrd >= 16) {
            if (j < 8)
                if (myMax < smem[i+8*WARP_SIZE]) {
                    smem[i] = myMax = smem[i+8*WARP_SIZE];
                    mark[i] = myMark = mark[i+8*WARP_SIZE];
                }
            __syncthreads();
        }
        if (num_thrd >= 8) {
            if (j < 4)
                if (myMax < smem[i+4*WARP_SIZE]) {
                    smem[i] = myMax = smem[i+4*WARP_SIZE];
                    mark[i] = myMark = mark[i+4*WARP_SIZE];
                }
            __syncthreads();
        }
        if (num_thrd >= 4) {
            if (j < 2)
                if (myMax < smem[i+2*WARP_SIZE]) {
                    smem[i] = myMax = smem[i+2*WARP_SIZE];
                    mark[i] = myMark = mark[i+2*WARP_SIZE];
                }
            __syncthreads();
        }
        if ( smem[threadIdx.x] > smem[threadIdx.x+WARP_SIZE] ) {
            myMax = smem[threadIdx.x];
            myMark = mark[threadIdx.x];
        }else {
            myMax = smem[threadIdx.x+WARP_SIZE];
            myMark = mark[threadIdx.x+WARP_SIZE];
        }

        i = blockIdx.x*WARP_SIZE + threadIdx.x;
        if (threadIdx.y == 0) res[i] = myMark;
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

template<class T>
__global__ void kWeightUpdate(T* x, T* y, T decay_rate, int ld, int fd) {
    int i = blockIdx.x*TILE_DIM + threadIdx.x;
    int j = blockIdx.y*TILE_DIM + threadIdx.y;

    if (i < ld - 1) {
        if (blockIdx.y < blockDim.y - 1) {
            for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS)
                x[i+(j+k)*ld] = x[i+(j+k)*ld]*decay_rate + y[i+(j+k)*ld];
        }else {
            for (int k = 0; j < fd - 1 && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) //TODO:change to k++
                x[i+j*ld] = x[i+j*ld]*decay_rate + y[i+j*ld];
        }
    }else if (i == ld - 1) {
        if (blockIdx.y < blockDim.y - 1) {
            for (int k = 0; k < TILE_DIM; k += BLOCK_ROWS)
                x[i+(j+k)*ld] = x[i+(j+k)*ld] + y[i+(j+k)*ld];
        }else {
            for (int k = 0; j < fd - 1 && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) 
                x[i+j*ld] = x[i+j*ld] + y[i+j*ld];
        }
    }
}


#endif //DISABLE_GPU
#endif //KERNELS_CUH
