//
// Created by hulei on 22-9-16.
//

#ifndef CUDATEST_LARGEUINT_CUH
#define CUDATEST_LARGEUINT_CUH

#include <cstdint>

template<uint32_t ElementNum>
class LargeUint{
public:
    constexpr static uint32_t N = ElementNum;
    __device__ LargeUint(){
        for (uint32_t i = 0; i < N; i++) {
            value[i] = 0;
        }
    }

    __device__ LargeUint(const uint64_t data[N]){
        for (uint32_t i = 0; i < N; i++) {
            value[i] = data[i];
        }
    }

    __host__ __device__ uint64_t& operator[](uint32_t i) {
        return value[i];
    }

    __host__ __device__ uint64_t operator[](uint32_t i) const{
        return value[i];
    }
private:
    uint64_t value[N];
};
#endif //CUDATEST_LARGEUINT_CUH
