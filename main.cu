#include <cuda_runtime.h>
#include <memory>

#define cudaCheckError(expr) {                                                               \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess) {                                                         \
        const char* error_str = cudaGetErrorString(e);                                       \
        printf("Cuda error:%s\n", error_str);                                                \
    }                                                                                        \
}

// Returns a * b + c + d, puts the carry in d
__device__ uint64_t mac(const uint64_t &a, const uint64_t &b, const uint64_t &c, uint64_t &carry) {
    uint64_t hi, lo;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
        "madc.hi.u64 %1, %2, %3, 0;\n\t"
        "add.cc.u64 %0, %0, %5;\n\t"
        "addc.u64 %1, %1, 0;\n\t"
            :"=l"(lo), "=l"(hi):"l"(a), "l"(b), "l"(c), "l"(carry));
    carry = hi;
    return lo;
}

template <uint32_t N>
__device__ void cal(uint64_t data[N]) {
    uint64_t t[N + 2]; //Wrong result
    //uint64_t t[N + 2] = {}; //Correct result

    for(uint32_t i = 0; i < N; i++) {
        uint64_t carry = 0;
        for(uint32_t j = 0; j < N; j++) {
            printf("%lu * %lu + %lu + %lu = ", data[j], data[j], t[j], carry);
            t[j] = mac(data[j], data[j], t[j], carry);
            printf("%lu with carry %lu\n", t[j], carry);
        }
    }
}

template <uint32_t N>
__global__
void kernel(uint64_t *data) {
    cal<N>(data);
}
int main(int argc, char **argv) {
    uint64_t *d_a;
    constexpr uint32_t N = 4;
    constexpr uint32_t COUNT = 1;
    constexpr uint32_t BYTES = N * COUNT * sizeof(uint64_t);
    cudaCheckError(cudaMalloc(&d_a, BYTES));
    uint64_t *h_a;
    h_a = new uint64_t[COUNT * N];
    for (int i = 0; i < COUNT * N; i++) {
        h_a[i] = i + 1;
    }
    cudaCheckError(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    kernel<4><<<1,1>>>(d_a);
    cudaCheckError(cudaMemcpy(h_a, d_a, BYTES, cudaMemcpyDeviceToHost));
    delete[] h_a;
    cudaCheckError(cudaFree(d_a));
}
