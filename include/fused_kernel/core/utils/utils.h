/*
   Some functions in this file are subject to other licenses

   Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <iostream>
#include <exception>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__ || __CUDACC_RTC__
#define FK_DEVICE_FUSE static constexpr __device__ __forceinline__
#define FK_DEVICE_CNST constexpr __device__ __forceinline__
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE __host__
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST __host__
#define FK_HOST_FUSE static inline __host__
#define FK_HOST_CNST inline constexpr __host__
#else
#define FK_HOST_DEVICE_FUSE static constexpr inline
#define FK_HOST_DEVICE_CNST constexpr inline
#define FK_HOST_FUSE static constexpr inline
#define FK_HOST_CNST constexpr inline
#endif 

#define CUDART_MAJOR_VERSION CUDART_VERSION/1000

using uchar = unsigned char;
using schar = signed char;
using uint = unsigned int;
using longlong = long long;
using ulonglong = unsigned long long;

using ushort = unsigned short;
using ulong = unsigned long;

namespace fk {
    constexpr inline void gpuAssert(cudaError_t code,
        const char *file,
        int line,
        bool abort = true) {
        if (code != cudaSuccess) {
            std::cout << "GPUassert: " << cudaGetErrorString(code) << " File: " << file << " Line:" << line << std::endl;
            if (abort) throw std::exception();
        }
    }
}

#define gpuErrchk(ans) { fk::gpuAssert((ans), __FILE__, __LINE__, true); }