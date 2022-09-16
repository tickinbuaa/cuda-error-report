//
// Created by hulei on 22-9-16.
//

#ifndef CUDATEST_MONTUINT_CUH
#define CUDATEST_MONTUINT_CUH

#include <stdio.h>
#include <curand_kernel.h>
#include "LargeUint.cuh"

//typedef unsigned __int128 uint128_t;

template<typename MC>
class MontUint{
public:
    using LU = LargeUint<4>;
    __device__ static MontUint from_normal(const LU &data) {
        LU value = data;
        to_mont(value);
        return value;
    }

    __device__ LU to_normal() const{
        LU data = value;
        mont_reduce(data);
        return data;
    }
protected:
    __device__ MontUint(const LU &data) {
        value = data;
    }
    
    static __device__ void to_mont(LU &data){
        mul_assign(data, MC::R2());
    }

    static __device__ void mont_reduce(LU &data) {
        uint64_t t[LU::N + 1];
        uint64_t carry = data[0];
        uint64_t m = MC::INV() * data[0];
        _mac_with_carry(m, MC::MODULO()[0], carry);
        for(uint32_t j = 1; j < LU::N; j++)
            t[j - 1] = _mac_with_carry(m, MC::MODULO()[j], data[j], carry);

        t[LU::N - 1] = carry;
        t[LU::N] = 0;
        for(uint32_t i = 1; i < LU::N; i++) {
            carry = 0;
            m = MC::INV() * t[0];
            _mac_with_carry(m, MC::MODULO()[0], t[0], carry);
            for(uint32_t j = 1; j < LU::N; j++)
                t[j - 1] = _mac_with_carry(m, MC::MODULO()[j], t[j], carry);

            t[LU::N - 1] = _add_with_carry(t[LU::N], carry);
            t[LU::N] = carry;
        }

        data = LU(t);
    }

    static __device__ void mul_assign(LU &a, const LU &b) {
        uint64_t t[LU::N + 2] = {};
        uint64_t carry = 0;
        for(uint32_t j = 0; j < LU::N; j++) {
            t[j] = _mac_with_carry(a[j], b[0], carry);
        }
        t[LU::N] = carry;
        carry = 0;
        uint64_t m = MC::INV() * t[0];
        _mac_with_carry(m, MC::MODULO()[0], t[0], carry);
        for(uint32_t j = 1; j < LU::N; j++)
            t[j - 1] = _mac_with_carry(m, MC::MODULO()[j], t[j], carry);

        t[LU::N - 1] = _add_with_carry(t[LU::N], carry);
        t[LU::N] = carry;

        for(uint32_t i = 1; i < LU::N; i++) {
            carry = 0;
            for(uint32_t j = 0; j < LU::N; j++) {
                t[j] = _mac_with_carry(a[j], b[i], t[j], carry);
            }
            t[LU::N] = _add_with_carry(t[LU::N], carry);
            t[LU::N + 1] = carry;
            carry = 0;
            m = MC::INV() * t[0];
            _mac_with_carry(m, MC::MODULO()[0], t[0], carry);
            for(uint32_t j = 1; j < LU::N; j++)
                t[j - 1] = _mac_with_carry(m, MC::MODULO()[j], t[j], carry);

            t[LU::N - 1] = _add_with_carry(t[LU::N], carry);
            t[LU::N] = t[LU::N + 1] + carry;
        }

        a = LU(t);
    }

    // Returns a * b + c + carry, puts the carry in carry
    static __device__ uint64_t _mac_with_carry(const uint64_t &a, const uint64_t &b, const uint64_t &c, uint64_t &carry) {
        uint64_t hi = 0, lo = 0;
        if (blockDim.x * blockIdx.x + threadIdx.x == 0){
            printf("%lu * %lu + %lu + %lu = ", a, b, c, carry);
        }
        asm("mad.lo.cc.u64 %0, %2, %3, %4;\n\t"
            "madc.hi.u64 %1, %2, %3, 0;\n\t"
            "add.cc.u64 %0, %0, %5;\n\t"
            "addc.u64 %1, %1, 0;\n\t"
                :"+l"(lo), "+l"(hi):"l"(a), "l"(b), "l"(c), "l"(carry));
        carry = hi;
        if (blockDim.x * blockIdx.x + threadIdx.x == 0){
            printf("%lu with carry %lu\n", lo, carry);
        }
        return lo;
    }

    // Returns a * b + carry, puts the carry in carry
    static __device__ uint64_t _mac_with_carry(const uint64_t &a, const uint64_t &b, uint64_t &carry) {
        uint64_t lo = 0;
        asm("mad.lo.cc.u64 %0, %2, %3, %1;\n\t"
            "madc.hi.u64 %1, %2, %3, 0;\n\t"
                :"=l"(lo), "+l"(carry):"l"(a), "l"(b));
        return lo;
    }

    // Returns a + b, return carry in b
    static __device__ uint64_t _add_with_carry(const uint64_t &a, uint64_t &b){
        uint64_t lo = 0;
        asm("add.cc.u64 %0, %1, %2;\n\t"
            "addc.u64 %1, 0, 0;\n\t"
                :"=l"(lo), "+l"(b): "l"(a));
        return lo;
    }
private:
    LU value;
};

#endif //CUDATEST_MONTUINT_CUH
