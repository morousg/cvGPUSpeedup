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

#define FK_DEVICE_FUSE static constexpr __device__ __forceinline__
#define FK_DEVICE_CNST constexpr __device__ __forceinline__
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE __host__
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST __host__
#define FK_HOST_FUSE static inline __host__
#define FK_HOST_CNST inline constexpr __host__

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

// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.

    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(uchar v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(schar v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(ushort v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(short v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(uint v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(int v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(float v) { return T(v); }
    template <typename T> constexpr __device__ __forceinline__ T saturate_cast(double v) { return T(v); }

    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(schar v)
    {
        uint res = 0;
        int vi = v;
        asm("cvt.sat.u8.s8 %0, %1;" : "=r"(res) : "r"(vi));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(short v)
    {
        uint res = 0;
        asm("cvt.sat.u8.s16 %0, %1;" : "=r"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(ushort v)
    {
        uint res = 0;
        asm("cvt.sat.u8.u16 %0, %1;" : "=r"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(int v)
    {
        uint res = 0;
        asm("cvt.sat.u8.s32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(uint v)
    {
        uint res = 0;
        asm("cvt.sat.u8.u32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(float v)
    {
        uint res = 0;
        asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(res) : "f"(v));
        return res;
    }
    template <> __device__ __forceinline__ uchar saturate_cast<uchar>(double v)
    {
        uint res = 0;
        asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(res) : "d"(v));
        return res;
    }

    template <> __device__ __forceinline__ schar saturate_cast<schar>(uchar v)
    {
        uint res = 0;
        uint vi = v;
        asm("cvt.sat.s8.u8 %0, %1;" : "=r"(res) : "r"(vi));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(short v)
    {
        uint res = 0;
        asm("cvt.sat.s8.s16 %0, %1;" : "=r"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(ushort v)
    {
        uint res = 0;
        asm("cvt.sat.s8.u16 %0, %1;" : "=r"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(int v)
    {
        uint res = 0;
        asm("cvt.sat.s8.s32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(uint v)
    {
        uint res = 0;
        asm("cvt.sat.s8.u32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(float v)
    {
        uint res = 0;
        asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(v));
        return res;
    }
    template <> __device__ __forceinline__ schar saturate_cast<schar>(double v)
    {
        uint res = 0;
        asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(res) : "d"(v));
        return res;
    }

    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(schar v)
    {
        ushort res = 0;
        int vi = v;
        asm("cvt.sat.u16.s8 %0, %1;" : "=h"(res) : "r"(vi));
        return res;
    }
    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(short v)
    {
        ushort res = 0;
        asm("cvt.sat.u16.s16 %0, %1;" : "=h"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(int v)
    {
        ushort res = 0;
        asm("cvt.sat.u16.s32 %0, %1;" : "=h"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(uint v)
    {
        ushort res = 0;
        asm("cvt.sat.u16.u32 %0, %1;" : "=h"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(float v)
    {
        ushort res = 0;
        asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(res) : "f"(v));
        return res;
    }
    template <> __device__ __forceinline__ ushort saturate_cast<ushort>(double v)
    {
        ushort res = 0;
        asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(res) : "d"(v));
        return res;
    }

    template <> __device__ __forceinline__ short saturate_cast<short>(ushort v)
    {
        short res = 0;
        asm("cvt.sat.s16.u16 %0, %1;" : "=h"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ short saturate_cast<short>(int v)
    {
        short res = 0;
        asm("cvt.sat.s16.s32 %0, %1;" : "=h"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ short saturate_cast<short>(uint v)
    {
        short res = 0;
        asm("cvt.sat.s16.u32 %0, %1;" : "=h"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ short saturate_cast<short>(float v)
    {
        short res = 0;
        asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(res) : "f"(v));
        return res;
    }
    template <> __device__ __forceinline__ short saturate_cast<short>(double v)
    {
        short res = 0;
        asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(res) : "d"(v));
        return res;
    }

    template <> __device__ __forceinline__ int saturate_cast<int>(uint v)
    {
        int res = 0;
        asm("cvt.sat.s32.u32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ int saturate_cast<int>(float v)
    {
        return __float2int_rn(v);
    }
    template <> __device__ __forceinline__ int saturate_cast<int>(double v)
    {
        return __double2int_rn(v);
    }

    template <> __device__ __forceinline__ uint saturate_cast<uint>(schar v)
    {
        uint res = 0;
        int vi = v;
        asm("cvt.sat.u32.s8 %0, %1;" : "=r"(res) : "r"(vi));
        return res;
    }
    template <> __device__ __forceinline__ uint saturate_cast<uint>(short v)
    {
        uint res = 0;
        asm("cvt.sat.u32.s16 %0, %1;" : "=r"(res) : "h"(v));
        return res;
    }
    template <> __device__ __forceinline__ uint saturate_cast<uint>(int v)
    {
        uint res = 0;
        asm("cvt.sat.u32.s32 %0, %1;" : "=r"(res) : "r"(v));
        return res;
    }
    template <> __device__ __forceinline__ uint saturate_cast<uint>(float v)
    {
        return __float2uint_rn(v);
    }
    template <> __device__ __forceinline__ uint saturate_cast<uint>(double v)
    {
        return __double2uint_rn(v);
    }
}

#define gpuErrchk(ans) { fk::gpuAssert((ans), __FILE__, __LINE__, true); }