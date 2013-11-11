#include <kernels.cuh>
#include <common.cuh>
#include <curand.h>
#include <curand_kernel.h>

__global__ void kSetupCurand(curandState *state, int nelem, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) curand_init(seed, i, 0, &state[i]);
}

