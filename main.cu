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

template<uint32_t ElementNum>
class LargeUint{
public:
    constexpr static uint32_t N = ElementNum;
    __device__ LargeUint(const uint64_t data[N]){
        for (uint32_t i = 0; i < N; i++) {
            value[i] = data[i];
        }
    }

    __device__ uint64_t& operator[](uint32_t i) {
        return value[i];
    }

    __device__ uint64_t operator[](uint32_t i) const{
        return value[i];
    }
private:
    uint64_t value[N];
};

template<typename LargeUint>
class Constant{
public:
    using LU = LargeUint;
    static __device__ LU constant();
};

class Constant6 : public Constant<LargeUint<6>>{
public:
    static __device__ LargeUint<6> constant() {
        const uint64_t value[6] = {1,2,3,4,5,6};
        return LargeUint<6>(value);
    }
};

template<typename C>
__device__ void cal(LargeUint<C::LU::N> data) {
    constexpr uint32_t N = C::LU::N;
    //uint64_t t[N + 2]; //Wrong result
    uint64_t t[N + 2] = {}; //Correct result
    uint64_t carry = 0;
//Cause wrong out put
#pragma unroll 1
    for(uint32_t j = 0; j < N; j++) {
        printf("%lx * %lx + %lx + %lx = ", data[j], C::constant()[j], t[j], carry);
        data[j] = mac(data[j], C::constant()[j], t[j], carry);
        printf("%lx with carry %lx\n", data[j], carry);
    }
}



template <typename C>
__global__
void kernel(uint64_t *data) {
    constexpr uint32_t N = C::LU::N;
    uint64_t value[N];
    for (uint32_t i = 0; i < N; i++){
        value[i] = data[i];
    }
    cal<C>(LargeUint<N>(value));
    for (uint32_t i = 0; i < N; i++) {
        data[i] = value[i];
    }
}

int main(int argc, char **argv) {
    uint64_t *d_a;
    constexpr uint32_t N = 6;
    constexpr uint32_t COUNT = 1;
    constexpr uint32_t BYTES = N * COUNT * sizeof(uint64_t);
    cudaCheckError(cudaMalloc(&d_a, BYTES));
    uint64_t *h_a;
    h_a = new uint64_t[COUNT * N];
    for (int i = 0; i < COUNT * N; i++) {
        h_a[i] = ~(uint64_t)0;
    }
    cudaCheckError(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    kernel<Constant6><<<1,1>>>(d_a);
    cudaCheckError(cudaMemcpy(h_a, d_a, BYTES, cudaMemcpyDeviceToHost));
    printf("%lx %lx %lx %lx %lx\n", h_a[0], h_a[1], h_a[2], h_a[3], h_a[4], h_a[5]);
    delete[] h_a;
    cudaCheckError(cudaFree(d_a));
}
