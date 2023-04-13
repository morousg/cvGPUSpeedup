/* 
   Some device functions are modifications of other libraries

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
#include "cuda_vector_utils.cuh"
#include "operation_types.cuh"
#include <cooperative_groups.h>
#include <vector>
#include <unordered_map>

namespace cg = cooperative_groups;

namespace fk {

struct Point {
  __device__ __forceinline__ __host__ Point(const uint x_, const uint y_) : x(x_), y(y_) {}
  __device__ __forceinline__ __host__ Point(){}
  uint x;
  uint y;  
};

inline uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
    const uint modX = width % blockDimx;
    const uint modY = height % blockDimy;
    const uint th_disabled_in_X = blockDimx - modX;
    const uint th_disabled_in_Y = blockDimy - modY;
    return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
}

template <uint bxS_t, uint byS_t>
struct computeBestSolution {};

template <uint bxS_t>
struct computeBestSolution<bxS_t, 0> {
    inline void operator()(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) const {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = bxS_t;
            byS = 0;
            if (minDiscardedThreads == 0) return; 
        }
        computeBestSolution<bxS_t, 1>()(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
    }
};

template <uint bxS_t>
struct computeBestSolution<bxS_t, 1> {
    inline void operator()(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) const {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = bxS_t;
            byS = 1;
            if (minDiscardedThreads == 0) return; 
        }
        computeBestSolution<bxS_t+1, 0>()(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
    }
};

template <>
struct computeBestSolution<3, 1> {
    inline void operator()(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint (&blockDimX)[4], const uint (&blockDimY)[2][4]) const {
        const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
        if (minDiscardedThreads > currentDiscardedThreads) {
            minDiscardedThreads = currentDiscardedThreads;
            bxS = 3;
            byS = 1;
        }
    }
};


inline dim3 getBlockSize(const uint width, const uint height, const uint planes = 1) {
    constexpr uint blockDimX[4]    = { 32, 64, 128, 256  };  // Possible block sizes in the x axis
    constexpr uint blockDimY[2][4] = {{ 8,  4,   2,   1},
                                      { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x
    
    uint minDiscardedThreads = UINT_MAX;
    uint bxS = 0; // from 0 to 3
    uint byS = 0; // from 0 to 1
    
    computeBestSolution<0, 0>()(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

    return dim3(blockDimX[bxS], blockDimY[byS][bxS], planes);
}

enum MemType { Device, Host, HostPinned };

template <typename T>
struct PtrAccessor {
    T* data;
    uint width;
    uint height;
    uint planes;
    uint pitch;
    dim3 adjusted_blockSize;
    MemType type;
    int deviceID;

    template <typename C>
    __host__ __device__ __forceinline__ 
    constexpr bool castIsOutOfBounds() const {
        return sizeInBytes() % sizeof(C) == 0;
    }

    __host__ __device__ __forceinline__ 
    constexpr uint sizeInBytes() const {
        return pitch * height;
    }

    __host__ __device__ __forceinline__ 
    constexpr bool hasPadding() const {
        return pitch != sizeof(T) * width;
    }

    __host__ __device__ __forceinline__ const T*__restrict__ at_cr(const Point p) const {
        return (const T*)((const char*)data + (p.y * pitch)) + p.x;
    }

    __host__ __device__ __forceinline__ const T* at_c(const Point p) const {
        return (const T*)((const char*)data + (p.y * pitch)) + p.x;
    }

    __host__ __device__ __forceinline__ T* at(const Point p) const {
        return (T*)((char*)data + (p.y * pitch)) + p.x;
    }

    __host__ __device__ __forceinline__ 
    constexpr uint getNumElements() const {
        return width * height;
    }
};

template <typename T>
class Ptr3D {

private:
    struct refPtr {
        void* ptr;
        int cnt;  
    };
    refPtr* ref{ nullptr };
    PtrAccessor<T> patterns;

    __host__ inline void freePrt() {
        if (ref) {
            ref->cnt--;
            if (ref->cnt == 0) {
                switch (patterns.type) {
                    case Device:
                        gpuErrchk(cudaFree(ref->ptr));
                        break;
                    case Host:
                        free(ref->ptr);
                        break;
                    case HostPinned:
                        gpuErrchk(cudaFreeHost(ref->ptr));
                        break;
                    default:
                        break;
                }
                free(ref);
            }
        }
    }

    __host__ inline Ptr3D(PtrAccessor<T> patterns_, refPtr * ref_) : patterns(patterns_), ref(ref_) {}

    __host__ inline void allocDevice() {
        int currentDevice;
        gpuErrchk(cudaGetDevice(&currentDevice));
        gpuErrchk(cudaSetDevice(patterns.deviceID));
        if (patterns.pitch == 0) {
            size_t pitch_temp;
            gpuErrchk(cudaMallocPitch(&patterns.data, &pitch_temp, sizeof(T) * patterns.width, patterns.height * patterns.planes));
            patterns.pitch = pitch_temp;
        } else {
            gpuErrchk(cudaMalloc(&patterns.data, patterns.pitch * patterns.height * patterns.planes));
        }
        if (currentDevice != patterns.deviceID) {
            gpuErrchk(cudaSetDevice(currentDevice));
        }
    }

    __host__ inline void allocHost() {
        patterns.data = (T*)malloc(sizeof(T) * patterns.width * patterns.height * patterns.planes);
        patterns.pitch = sizeof(T) * patterns.width; //Here we don't support padding
    }

    __host__ inline void allocHostPinned() {
        gpuErrchk(cudaMallocHost(&patterns.data, sizeof(T) * patterns.width * patterns.height * patterns.planes));
        patterns.pitch = sizeof(T) * patterns.width; //Here we don't support padding
    }

public:

    __host__ inline Ptr3D() {}

    __host__ inline Ptr3D(const Ptr3D<T>& other) {
        patterns = other.patterns;
        if (other.ref) {
            ref = other.ref;
            ref->cnt++;
        }
    }

    __host__ inline Ptr3D(uint width_, uint height_, uint pitch_ = 0, uint planes_ = 1, MemType type_ = Device, int deviceID_ = 0) {
        allocPtr(width_, height_, pitch_, planes_, type_, deviceID_);
    }

    __host__ inline Ptr3D(T * data_, uint width_, uint height_, uint pitch_, uint planes_ = 1, MemType type_ = Device, int deviceID_ = 0) {
        patterns.data = data_;
        patterns.width = width_;
        patterns.height = height_;
        patterns.pitch = pitch_;
        patterns.planes = planes_;
        patterns.type = type_;
        patterns.deviceID = deviceID_;
        patterns.adjusted_blockSize = fk::getBlockSize(patterns.width, patterns.height);
    }

    __host__ inline ~Ptr3D() {
        // TODO: add gpuCkeck
        freePrt();
    }

    __host__ inline PtrAccessor<T> d_ptr() const { return patterns; }

    __host__ inline operator PtrAccessor<T>() const { return patterns; }

    __host__ inline void allocPtr(const uint width_, const uint height_, const uint pitch_ = 0, const uint planes_ = 1, const MemType type_ = Device, const int deviceID_ = 0) {
        patterns.width = width_;
        patterns.height = height_;
        patterns.pitch = pitch_;
        patterns.planes = planes_;
        patterns.type = type_;
        patterns.deviceID = deviceID_;
        ref = (refPtr*)malloc(sizeof(refPtr));
        ref->cnt = 1;

        switch (type_) {
            case Device:
                allocDevice();
                patterns.adjusted_blockSize = fk::getBlockSize(patterns.width, patterns.height);
                break;
            case Host:
                allocHost();
                break;
            case HostPinned:
                allocHostPinned();
                break;
            default:
                break;
        }

        ref->ptr = patterns.data;
    }

    __host__ inline Ptr3D<T> crop(Point p, uint width_n, uint height_n, uint planes_n = 1) {
        T* ptr = patterns.at(p);
        ref->cnt++;
        const PtrAccessor<T> newAccessor = {ptr, width_n, height_n, planes_n, patterns.pitch,
                                            fk::getBlockSize(width_n, height_n), patterns.type, patterns.deviceID};
        return {newAccessor, ref};
    }

    __host__ inline uint width() const {
        return patterns.width;
    }
    __host__ inline uint height() const {
        return patterns.height;
    }
    __host__ inline uint pitch() const {
        return patterns.pitch;
    }
    __host__ inline T* data() const {
        return patterns.data;
    }

    __host__ inline dim3 getBlockSize() const {
        return patterns.adjusted_blockSize;
    }
};

template <typename I, typename O=I>
struct perthread_write_2D {
    __device__ __forceinline__ void operator()(const dim3 thread, I input, PtrAccessor<O> output) {
        *(output.at({thread.x, thread.y})) = input;
    }
};

template <typename I, typename Enabler=void>
struct perthread_split_write_2D;

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<CN(I) == 2>> {
    __device__ __forceinline__ void operator()(const dim3 thread, I input,
                               PtrAccessor<decltype(I::x)> output1,
                               PtrAccessor<decltype(I::y)> output2) {
        const Point p(thread.x, thread.y);
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y;
    }
};

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<CN(I) == 3>> {
    __device__ __forceinline__ void operator()(const dim3 thread, I input, 
                               PtrAccessor<decltype(I::x)> output1, 
                               PtrAccessor<decltype(I::y)> output2,
                               PtrAccessor<decltype(I::z)> output3) {
        const Point p(thread.x, thread.y);
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y; 
        *output3.at(p) = input.z;
    }
};

template <typename I>
struct perthread_split_write_2D<I, typename std::enable_if_t<CN(I) == 4>> {
    __device__ __forceinline__ void operator()(const dim3 thread, I input, 
                               PtrAccessor<decltype(I::x)> output1, 
                               PtrAccessor<decltype(I::y)> output2,
                               PtrAccessor<decltype(I::z)> output3,
                               PtrAccessor<decltype(I::w)> output4) { 
        const Point p(thread.x, thread.y);
        *output1.at(p) = input.x; 
        *output2.at(p) = input.y; 
        *output3.at(p) = input.z;
        *output4.at(p) = input.w; 
    }
};

// The following code is a modification of the OpenCV file resize.cu
// which has the following license

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//
//M*/

//! interpolation algorithm
enum InterpolationType {
    /** nearest neighbor interpolation */
    INTER_NEAREST        = 0,
    /** bilinear interpolation */
    INTER_LINEAR         = 1,
    /** bicubic interpolation */
    INTER_CUBIC          = 2,
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    INTER_AREA           = 3,
    /** Lanczos interpolation over 8x8 neighborhood */
    INTER_LANCZOS4       = 4,
    /** Bit exact bilinear interpolation */
    INTER_LINEAR_EXACT = 5,
    /** Bit exact nearest neighbor interpolation. This will produce same results as
    the nearest neighbor method in PIL, scikit-image or Matlab. */
    INTER_NEAREST_EXACT  = 6,
    /** mask for interpolation codes */
    INTER_MAX            = 7,
    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    source image, they are set to zero */
    WARP_FILL_OUTLIERS   = 8,
    /** flag, inverse transformation

    For example, #linearPolar or #logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    WARP_INVERSE_MAP     = 16,
    NONE = 17
};

template <typename T, InterpolationType INTER_T>
struct interpolate_read;

template <typename T>
struct interpolate_read<T, InterpolationType::INTER_LINEAR> {
    __device__ __forceinline__ T operator()(const PtrAccessor<T> input, const float fy, const float fx, const int dst_x, const int dst_y) {
        const float src_x = dst_x * fx;
        const float src_y = dst_y * fy;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;        
        const int x2_read = ::min(x2, input.width - 1);
        const int y2_read = ::min(y2, input.height - 1);

        using floatcn_t = typename VectorType<float, VectorTraits<T>::cn>::type;
        floatcn_t out = make_set<floatcn_t>(0.f);
        T src_reg = *input.at_cr(Point(x1, y1));
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *input.at_cr(Point(x2_read, y1));
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *input.at_cr(Point(x1, y2_read));
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *input.at_cr(Point(x2_read, y2_read));
        out = out + src_reg * ((src_x - x1) * (src_y - y1));
        
        return saturate_cast<T>(out);
    } 
};

}
