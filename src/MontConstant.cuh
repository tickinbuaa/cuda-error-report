//
// Created by hulei on 22-9-16.
//

#ifndef CUDATEST_MONTCONSTANT_CUH
#define CUDATEST_MONTCONSTANT_CUH
#include <cstdint>

class MontConstant{
public:
    using LU = LargeUint<4>;
    static __device__ LU R2() {
        const uint64_t value[4] = {0xc96cefdf08fac539ULL, 0x349f6a083baaf1ULL, 0xccb2517e0a2c7bb5ULL, 0x2b4a05e9c97ULL};
        return value;
    }
    static __device__ LU MODULO() {
        const uint64_t value[4] = {0x939ee740b0572bf1ULL, 0x5fcd6d15634c091dULL, 0x64c242a72893e769ULL, 0x6812b683558ULL};
        return value;
    }
    static __device__ uint64_t INV() {
        return 0x5638cb0e5c1f9aefULL;
    }
};

#endif //CUDATEST_MONTCONSTANT_CUH
