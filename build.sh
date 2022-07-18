#!/bin/sh
mkdir -p release-build
cmake -S . -B release-build
cmake --build release-build
