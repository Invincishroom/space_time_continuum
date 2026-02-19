#!/bin/bash

mkdir -p build/release
cd build/release
cmake -DEstimator=ON -DDebug=OFF ../.. || exit 1
cmake --build . -j${nproc} || exit 1
cd ../..