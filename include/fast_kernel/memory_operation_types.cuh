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
#include "ptr_nd.cuh"
#include <cooperative_groups.h>
#include <vector>
#include <unordered_map>

namespace cg = cooperative_groups;

namespace fk {

template <ND D, typename T>
struct perthread_write {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<D,T>& output) {
        *PtrAccessor<D>::point(thread, output) = input;
    }
};

template <typename T, typename Enabler=void>
struct perthread_tensor_split_write;

template <typename T>
struct perthread_tensor_split_write<T, typename std::enable_if_t<CN(T) == 2>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
    }
};

template <typename T>
struct perthread_tensor_split_write<T, typename std::enable_if_t<CN(T) == 3>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.z;
    }
};

template <typename T>
struct perthread_tensor_split_write<T, typename std::enable_if_t<CN(T) == 4>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        using BaseType = typename VectorTraits<T>::base;
        BaseType* work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.y;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.z;
        work_plane += (ptr.dims.width * ptr.dims.height);
        *work_plane = input.w;
    }
};

template <ND D, typename T, typename Enabler=void>
struct perthread_split_write;

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 2>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<D,decltype(T::x)>& output1,
                             const RawPtr<D,decltype(T::y)>& output2) {
        *PtrAccessor<D>::point(thread, output1) = input.x;
        *PtrAccessor<D>::point(thread, output2) = input.y;
    }
};

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 3>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, 
                             const RawPtr<D,decltype(T::x)>& output1, 
                             const RawPtr<D,decltype(T::y)>& output2,
                             const RawPtr<D,decltype(T::z)>& output3) {
        *PtrAccessor<D>::point(thread, output1) = input.x;
        *PtrAccessor<D>::point(thread, output2) = input.y;
        *PtrAccessor<D>::point(thread, output3) = input.z;
    }
};

template <ND D, typename T>
struct perthread_split_write<D, T, typename std::enable_if_t<CN(T) == 4>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, 
                               const RawPtr<D,decltype(T::x)>& output1, 
                               const RawPtr<D,decltype(T::y)>& output2,
                               const RawPtr<D,decltype(T::z)>& output3,
                               const RawPtr<D,decltype(T::w)>& output4) { 
        *PtrAccessor<D>::point(thread, output1) = input.x;
        *PtrAccessor<D>::point(thread, output2) = input.y;
        *PtrAccessor<D>::point(thread, output3) = input.z;
        *PtrAccessor<D>::point(thread, output4) = input.w;
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

template <ND D, typename I, InterpolationType INTER_T, int NPtr>
struct interpolate_read;

template <typename I>
struct interpolate_read<_2D, I, InterpolationType::INTER_LINEAR, 1> {
    static __device__ __forceinline__ const I exec(const Point& thread, const RawPtr<_2D,I>& ptr,
                                                   const float& fx, const float& fy) {
        const float src_x = thread.x * fx;
        const float src_y = thread.y * fy;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;        
        const int x2_read = ::min(x2, ptr.dims.width - 1);
        const int y2_read = ::min(y2, ptr.dims.height - 1);

        using floatcn_t = typename VectorType<float, VectorTraits<I>::cn>::type;
        floatcn_t out = make_set<floatcn_t>(0.f);
        I src_reg = *PtrAccessor<_2D>::cr_point(Point(x1, y1), ptr);
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(Point(x2_read, y1), ptr);
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(Point(x1, y2_read), ptr);
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *PtrAccessor<_2D>::cr_point(Point(x2_read, y2_read), ptr);
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        return saturate_cast<I>(out);
    }
};

template <typename I, int NPtr>
struct interpolate_read<_3D, I, InterpolationType::INTER_LINEAR, NPtr> {
    FK_DEVICE_FUSE const I exec(const Point& thread,
                          const RawPtr<_2D,I> (&ptr)[NPtr], 
                          const float (&fx)[NPtr], const float (&fy)[NPtr]) {
        return interpolate_read<_2D, I, InterpolationType::INTER_LINEAR, 1>
                                ::exec(thread, ptr[thread.z], fx[thread.z], fy[thread.z]);

    }
};

}
