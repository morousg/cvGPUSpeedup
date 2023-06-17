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
#include "operations.cuh"
#include "ptr_nd.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace fk {

template <ND D, typename T>
struct PerThreadRead {
    FK_DEVICE_FUSE T exec(const Point& thread, const RawPtr<D,T>& ptr) {
        return *PtrAccessor<D>::cr_point(thread, ptr);
    }
    using Type = T;
    using ParamsType = RawPtr<D,T>;
};

template <ND D, typename T>
struct PerThreadWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<D,T>& output) {
        *PtrAccessor<D>::point(thread, output) = input;
    }
    using Type = T;
    using ParamsType = RawPtr<D,T>;
};

template <typename T>
struct TensorRead {
    FK_DEVICE_FUSE T exec(const Point& thread, const RawPtr<_3D, T>& ptr) {
        return *PtrAccessor<_3D>::cr_point(thread, ptr);
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
};

template <typename T>
struct TensorWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const RawPtr<_3D, T>& output) {
        *PtrAccessor<_3D>::point(thread, output) = input;
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, T>;
};


template <typename T>
struct TensorSplitWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        static_assert(cn<T> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        typename VectorTraits<T>::base* const work_plane = PtrAccessor<_3D>::point(thread, ptr);
        *work_plane = input.x;
        *(work_plane + planePixels) = input.y;
        if constexpr (cn<T> >= 3) {
            *(work_plane + (planePixels * 2)) = input.z;
        }
        if constexpr (cn<T> == 4) {
            *(work_plane + (planePixels * 3)) = input.w;
        }
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
};

template <typename T>
struct TensorSplitRead {
    FK_DEVICE_FUSE T exec(const Point& thread,
                          const RawPtr<_3D, typename VectorTraits<T>::base>& ptr) {
        static_assert(cn<T> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

        const int planePixels = ptr.dims.width * ptr.dims.height;

        const typename VectorTraits<T>::base* const work_plane = PtrAccessor<_3D>::cr_point(thread, ptr);
        if constexpr (cn<T> == 2) {
            return make_<T>(*work_plane, *(work_plane + planePixels));
        } else if constexpr (cn<T> == 3) {
            return make_<T>(*work_plane, *(work_plane + planePixels),
                            *(work_plane + (planePixels * 2)));
        } else {
            return make_<T>(*work_plane,
                            *(work_plane + planePixels),
                            *(work_plane + (planePixels * 2)),
                            *(work_plane + (planePixels * 3)));
        }
    }
    using Type = T;
    using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
};

template <ND D, typename T, typename Enabler=void>
struct SplitWriteParams {};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 2>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 3>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
};

template <ND D, typename T>
struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 4>> {
    RawPtr<D, decltype(T::x)> x;
    RawPtr<D, decltype(T::y)> y;
    RawPtr<D, decltype(T::z)> z;
    RawPtr<D, decltype(T::w)> w;
};

template <ND D, typename T>
struct SplitWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& input,
                             const SplitWriteParams<D, T>& params) {
        static_assert(cn<T> >= 2, "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
        *PtrAccessor<D>::point(thread, params.x) = input.x;
        *PtrAccessor<D>::point(thread, params.y) = input.y;
        if constexpr (cn<T> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
        if constexpr (cn<T> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
    }
    using Type = T;
    using ParamsType = SplitWriteParams<D, T>;
};

template <typename Operation, int NPtr>
struct BatchRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
                                                       const typename Operation::ParamsType (&params)[NPtr]) {
        return Operation::exec(thread, params[thread.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
};

template <typename Operation, int NPtr>
struct BatchWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
                             const typename Operation::ParamsType (&params)[NPtr]) {
        Operation::exec(thread, input, params[thread.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = typename Operation::ParamsType[NPtr];
};

/* The following code has the following copy right
 
   Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huget)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

template <typename ParamsType>
struct CircularMemoryParams {
    int first;
    ParamsType params;
};

template <typename Operation, int BATCH>
struct CircularBatchRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
        const CircularMemoryParams<typename Operation::ParamsType[BATCH]>& c_params) {
        const int fst = c_params.first;
        const Point newThreadIdx{ thread.x, thread.y, thread.z >= fst ? thread.z - fst : thread.z + (BATCH - fst) };
        return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
};

template <typename Operation, int BATCH>
struct CircularBatchWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
        const CircularMemoryParams<typename Operation::ParamsType[BATCH]>& c_params) {
        const int fst = c_params.first;
        const Point newThreadIdx{ thread.x, thread.y, thread.z >= fst ? thread.z - fst : thread.z + (BATCH - fst) };
        Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
};

template <typename Operation, int BATCH>
struct CircularTensorRead {
    FK_DEVICE_FUSE const typename Operation::Type exec(const Point& thread,
        const CircularMemoryParams<typename Operation::ParamsType>& c_params) {
        const int fst = c_params.first;
        const Point newThreadIdx{ thread.x, thread.y, thread.z >= fst ? thread.z - fst : thread.z + (BATCH - fst) };
        return Operation::exec(newThreadIdx, c_params.params);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
};

template <typename Operation, int BATCH>
struct CircularTensorWrite {
    FK_DEVICE_FUSE void exec(const Point& thread, const typename Operation::Type& input,
        const CircularMemoryParams<typename Operation::ParamsType>& c_params) {
        const int fst = c_params.first;
        const Point newThreadIdx{ thread.x, thread.y, thread.z >= fst ? thread.z - fst : thread.z + (BATCH - fst) };
        Operation::exec(newThreadIdx, input, c_params.params);
    }
    using Type = typename Operation::Type;
    using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
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

template <typename I>
struct InterpolateParams {
    RawPtr<_2D,I> ptr;
    float fx;
    float fy;
};

template <typename I, InterpolationType INTER_T>
struct InterpolateRead;

template <typename I>
struct InterpolateRead<I, InterpolationType::INTER_LINEAR> {
    static __device__ __forceinline__ const I exec(const Point& thread, const InterpolateParams<I>& params) {
        const RawPtr<_2D, I> ptr = params.ptr;

        const float src_x = thread.x * params.fx;
        const float src_y = thread.y * params.fy;

        const int x1 = __float2int_rd(src_x);
        const int y1 = __float2int_rd(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;        
        const int x2_read = ::min(x2, ptr.dims.width - 1);
        const int y2_read = ::min(y2, ptr.dims.height - 1);

        using floatcn_t = typename VectorType<float, cn<I>>::type;
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
    using Type = I;
    using ParamsType = InterpolateParams<I>;
};

}
