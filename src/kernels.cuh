#ifndef KERNELS_CUH
#define KERNELS_CUH
#ifndef DISABLE_GPU

#include <common.cuh>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <DOperators.cuh>


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
            for (int k = 0; (even_n || j < n) && k < TILE_DIM; k += BLOCK_ROWS, j += BLOCK_ROWS) {
                x[i+j*ldx] = op(x[i+j*ldx], y[i+j*ldy]);
            }
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


template<class T, int num_thrd, class Op, class OpTrans>
__device__ T dBatchReduce(Op op, OpTrans opTrans, T *x, T *smem, int *mark, int i, int j, int ld, int fd, int &ind) {
    T res = op.Unit;
    int myInd = 0;
    int k = threadIdx.y;
    const int n = ld*fd;
    const int blockSize = ld*num_thrd;
    while (j < n) {
        res = op(res, myInd, opTrans(x[j]), k, myInd);
        k += num_thrd;
        j += blockSize;
    }
    j = threadIdx.y;
    i = j*WARP_SIZE + threadIdx.x;
    smem[i] = res;
    mark[i] = myInd;
    __syncthreads();
    if (num_thrd >= 32) {
        const int step = 16;
        if (j < step) {
            smem[i] = res = op(res, myInd, smem[i+step*WARP_SIZE], mark[i+step*WARP_SIZE], myInd);
            mark[i] = myInd;
        }
        __syncthreads();
    }
    if (num_thrd >= 16) {
        const int step = 8;
        if (j < step) {
            smem[i] = res = op(res, myInd, smem[i+step*WARP_SIZE], mark[i+step*WARP_SIZE], myInd);
            mark[i] = myInd;
        }
        __syncthreads();
    }
    if (num_thrd >= 8) {
        const int step = 4;
        if (j < step) {
            smem[i] = res = op(res, myInd, smem[i+step*WARP_SIZE], mark[i+step*WARP_SIZE], myInd);
            mark[i] = myInd;
        }
        __syncthreads();
    }
    if (num_thrd >= 4) {
        const int step = 2;
        if (j < step) {
            smem[i] = res = op(res, myInd, smem[i+step*WARP_SIZE], mark[i+step*WARP_SIZE], myInd);
            mark[i] = myInd;
        }
        __syncthreads();
    }
    res = op(smem[threadIdx.x], mark[threadIdx.x], smem[threadIdx.x + WARP_SIZE], mark[threadIdx.x + WARP_SIZE], myInd);
    ind = myInd;
    return res;
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
        int myMark;
        T myMax = dBatchReduce<T, num_thrd, OpMaxReduce<T>, OpNop<T> >(OpMaxReduce<T>(), OpNop<T>(), drv, smem, mark, i, j, ld, fd, myMark);
        if (threadIdx.y == 0) res[i] = myMark;

        T mySum = dBatchReduce<T, num_thrd, OpSumReduce<T>, OpSubExp<T> >(OpSumReduce<T>(), OpSubExp<T>(myMax), drv, smem, mark, i, j, ld, fd, myMark);

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

template<class T, int num_thrd>
__global__ void kUnitLength(T *x, T *y, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    __shared__ int mark[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y*ld + i;
    const int blockSize = ld*num_thrd;

    int n = ld*fd;
    if (i < ld) {
        int myMark;
        T mySum = dBatchReduce<T, num_thrd, OpSumReduce<T>, OpSqr<T> >(OpSumReduce<T>(), OpSqr<T>(), x, smem, mark, i, j, ld, fd, myMark);
        mySum = sqrt(mySum);

        while (j < n) {
            y[j] = x[j]/mySum;
            j += blockSize;
        }
    }

}

template<class T>
__global__ void UnitLength(DMatrix<T> *x, DMatrix<T> *y, int fd) {
    dim3 grid((x->ld()-1)/WARP_SIZE+1, 1, 1);
    dim3 block(WARP_SIZE, 32, 1);
    kUnitLength<T,32><<<grid, block>>>(x->dev_data(), y->dev_data(), x->ld(), fd);
    CUDA_KERNEL_CHECK();
}

template<class T, int num_thrd>
__global__ void kArgmax(T *x, T *ind, T *res, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    __shared__ int mark[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y*ld + i;
    const int blockSize = ld*num_thrd;

    int n = ld*fd;
    if (i < ld) {
        int myMark;
        res[i] = dBatchReduce<T, num_thrd, OpMinReduce<T>, OpNop<T> >(OpMaxReduce<T>(), OpNop<T>(), x, smem, mark, i, j, ld, fd, myMark);
        ind[i] = myMark;
    }

}

template<class T>
__global__ void Argmax(DMatrix<T> *x, DMatrix<T> *ind, DMatrix<T> *res, int fd) {
    dim3 grid((x->ld()-1)/WARP_SIZE+1, 1, 1);
    dim3 block(WARP_SIZE, 32, 1);
    kArgmax<T,32><<<grid, block>>>(x->dev_data(), ind->dev_data(), res->dev_data(), x->ld(), fd);
    CUDA_KERNEL_CHECK();
}
#endif //DISABLE_GPU
#endif //KERNELS_CUH
