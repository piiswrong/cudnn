#ifndef KERNELS_CUH
#define KERNELS_CUH


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


#endif //KERNELS_CUH
