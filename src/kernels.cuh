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


template<class T, bool trans, int num_thrd, class Op, class OpTrans>
__device__ T dBatchReduce(Op op, OpTrans opTrans, T *x, T *smem, int *mark, int ld, int fd, int &ind) {
    T res = op.Unit;
    int myInd = 0;
    int n;
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y;
    if (trans) 
        n = ld;
    else 
        n = fd;
    for (;j < n; j += num_thrd) {
        if (trans)
            res = op(res, myInd, opTrans(x[j+i*ld]), j, myInd);
        else 
            res = op(res, myInd, opTrans(x[i+j*ld]), j, myInd);
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
        T myMax = dBatchReduce<T, false, num_thrd, OpMaxReduce<T>, OpNop<T> >(OpMaxReduce<T>(), OpNop<T>(), drv, smem, mark, ld, fd, myMark);
        if (threadIdx.y == 0) res[i] = myMark;

        T mySum = dBatchReduce<T, false, num_thrd, OpSumReduce<T>, OpSubExp<T> >(OpSumReduce<T>(), OpSubExp<T>(myMax), drv, smem, mark, ld, fd, myMark);

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

template<class T, bool trans, int num_thrd, class OpElem, class OpReduce, class OpAll, class OpNorm>
__global__ void kNormalize(OpElem opElem, OpReduce opReduce, OpAll opAll, OpNorm opNorm, T *x, T *y, T *norm, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    __shared__ int mark[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;
    int j = threadIdx.y*ld + i;
    const int blockSize = ld*num_thrd;

    if (i < ld) {
        int myMark;
        T mySum = dBatchReduce<T, trans, num_thrd, OpReduce, OpElem>(opReduce, opElem, x, smem, mark, ld, fd, myMark);
        mySum = opAll(mySum);
        if (norm != NULL && threadIdx.y == 0) {
            norm[i] = mySum;
        }

        if (trans) {
            for (j = threadIdx.y; j < ld; j += num_thrd) {
                y[j + i*ld] = opNorm(x[j + i*ld], mySum);
            }
        }else {
            int n = ld*fd;
            while (j < n) {
                y[j] = opNorm(x[j], mySum);
                j += blockSize;
            }
        }
    }
}



template<class T, int num_thrd>
__global__ void kArgmax(T *x, int *ind, T *res, int ld, int fd) {
    __shared__ T smem[WARP_SIZE*num_thrd];
    __shared__ int mark[WARP_SIZE*num_thrd];
    int i = blockIdx.x*WARP_SIZE + threadIdx.x;

    if (i < ld) {
        int myMark = 0;
        T myMax = dBatchReduce<T, false, num_thrd, OpMaxReduce<T>, OpNop<T> >(OpMaxReduce<T>(), OpNop<T>(), x, smem, mark, ld, fd, myMark);
        if (threadIdx.y == 0) {
            ind[i] = myMark;
            res[i] = myMax;
        }
    }

}



template<class T>
__global__ void kCluterNeuronDelta(T *scale, T *y, T *margin, T *res, int *index, T lambda) {
    int i = threadIdx.x;
    T bj = margin[index[i]];
    scale[i] = y[i] * (bj > res[i]) * -1.0 + lambda*(1-y[i]) * (res[i] > bj);
}


template<class T>
__global__ void kCluterNeuronAcc(T *acc, T *y, T *margin, T *res, int *index) {
    int i = threadIdx.x;
    T bj = margin[index[i]];
    acc[i] = y[i] * (bj > res[i]) + (1-y[i]) * (res[i] > bj);
}


template<class T> 
__global__ void kCluterNeuronBprop(T *delta, T *act, T *centers, int *index, T *res, T *scale, T *norm, int ld, int fd) {
    __shared__ T s_centers[TILE_DIM][TILE_DIM+1];
    __shared__ int s_index[BLOCK_ROWS];
    __shared__ T s_scale[TILE_DIM];
    __shared__ T s_norm[TILE_DIM];
    __shared__ T s_res[TILE_DIM];
    int i = blockIdx.x*TILE_DIM + threadIdx.y;
    int j = blockIdx.y*TILE_DIM + threadIdx.x;
    for (int k = 0; k < TILE_DIM && i < ld; k += BLOCK_ROWS, i += BLOCK_ROWS) {
        if (threadIdx.x == 0) s_index[threadIdx.y] = index[i];
        __syncthreads();
        s_centers[threadIdx.y + k][threadIdx.x] = centers[j + s_index[threadIdx.y]*fd];
    }
    i = blockIdx.x*TILE_DIM + threadIdx.x;
    j = blockIdx.y*TILE_DIM + threadIdx.y;
    if (threadIdx.y == 0) {
        s_scale[threadIdx.x] = scale[i];
        s_norm[threadIdx.x] = norm[i];
        s_res[threadIdx.x] = res[i];
    }
    __syncthreads();
    for (int k = 0; k < TILE_DIM && j < fd; k += BLOCK_ROWS, j += BLOCK_ROWS) {
        T n = s_norm[threadIdx.x];
        delta[i + j*ld] = s_scale[threadIdx.x]*(s_centers[threadIdx.x][threadIdx.y + k] - s_res[threadIdx.x]*act[i + j*ld]/n)/n;
    }
}

template<class T>
__global__ void kCluterNeuronReverseIndex(T *rindex, T *scale, T *index, int ld, int fd) {
    int i = threadIdx.x;
    rindex[index[i] + ld*i] = scale[i];
}

template<class T, class Dist>
__global__ void kComputeDistanceKernel(Dist dist, T *x, T *y, T *z, int ldx, int ldy, int ldz, int p) {
    __shared__ T sx[TILE_DIM][TILE_DIM];
    __shared__ T sy[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int i = bx*TILE_DIM + tx;
    int j = by*TILE_DIM + ty;

    T c = 0.0;
    for (int k = 0; k < p; k += TILE_DIM) {
        sx[tx][ty] = x[i + (k+threadIdx.y)*ldx];
        sy[tx][ty] = y[k + threadIdx.x + j*ldy];
        __syncthreads();

        for (int kk = 0; kk < TILE_DIM; kk++) {
            c += dist(sx[tx][kk], sy[kk][ty]);
        }
        __syncthreads();
    }
    z[i+j*ldz] = c;
}

#endif //DISABLE_GPU
#endif //KERNELS_CUH
