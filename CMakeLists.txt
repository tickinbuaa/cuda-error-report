cmake_minimum_required(VERSION 3.10)
project(CudaTest CUDA CXX)

include(FetchContent)
set(CMAKE_CUDA_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(CudaTest main.cu)

target_compile_options(CudaTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 -lineinfo
        >)

