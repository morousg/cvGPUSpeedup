#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

using uchar = unsigned char;

inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " File: " << file << " Line:" << line << std::endl;
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }