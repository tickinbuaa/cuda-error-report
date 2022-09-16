#!/bin/sh
mkdir -p build-release
cmake -S . -B build-release -DCMAKE_CUDA_ARCHITECTURES="75;86"
cmake --build build-release
