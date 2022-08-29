#!/bin/sh
mkdir -p build-release
cmake -S . -B build-release
cmake --build build-release
