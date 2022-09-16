#include <cuda_runtime.h>
#include <memory>
#include "MontUint.cuh"
#include "MontConstant.cuh"

#define cudaCheckError(expr) {                                                               \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess) {                                                         \
        const char* error_str = cudaGetErrorString(e);                                       \
        printf("Cuda error:%s\n", error_str);                                                \
    }                                                                                        \
}


__global__
void kernel(uint64_t *data) {
    using MU = MontUint<MontConstant>;
    MU a = MU::from_normal(data);
    MU::LU b = a.to_normal();
    for (uint32_t i = 0; i < MU::LU::N; i++) {
        data[i] = b[i];
    }
}

int main(int argc, char **argv) {
    uint64_t *d_a;
    constexpr uint32_t N = 4;
    cudaCheckError(cudaMalloc(&d_a, N * sizeof(uint64_t)));
    uint64_t h_a[4] = {14743922779321530014ULL, 4079684056696350560ULL, 3051051747104855183ULL, 481250544501ULL};
    cudaCheckError(cudaMemcpy(d_a, h_a, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    kernel<<<1,1>>>(d_a);
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaFree(d_a));
}
