#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

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