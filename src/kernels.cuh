#ifndef KERNELS_CUH
#define KERNELS_CUH


template<class T, class Op>
__global__ void kApplyBinaryOp(const T* dest, const T* x, int nelem, Op op) {
    const uint i= blockIdx.x * blockDim.x + threadIdx.x;
    dest[i] = op(dest[i], x[i]);
}



#endif //KERNELS_CUH
