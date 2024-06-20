/* Copyright 2023 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/operations.cuh>

#include <cuda.h>

namespace fk {
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

    // End of the following copyrights
    // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
    // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
    // Copyright (C) 2013, OpenCV Foundation, all rights reserved.

    namespace vec_math_detail
    {
        template <int cn, typename VecD> struct SatCastHelper;
        template <typename VecD> struct SatCastHelper<1, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x));
            }
        };
        template <typename VecD> struct SatCastHelper<2, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y));
            }
        };
        template <typename VecD> struct SatCastHelper<3, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z));
            }
        };
        template <typename VecD> struct SatCastHelper<4, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z), saturate_cast<D>(v.w));
            }
        };

        template <typename VecD, typename VecS> static __device__ __forceinline__ constexpr VecD saturate_cast_helper(const VecS& v)
        {
            return SatCastHelper<cn<VecD>, VecD>::cast(v);
        }
    }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template <typename I, typename O>
    struct SaturateCast {
        UNARY_DECL_EXEC(I, O) {
            return saturate_cast<OutputType>(input);
        }
    };

    struct SaturateFloatBase {
        UNARY_DECL_EXEC(float, float) {
            return Max<float>::exec(0.f, Min<float>::exec(input, 1.f));
        }
    };

    template <typename T>
    struct Saturate {
        using InputType = T;
        using OutputType = T;
        using ParamsType = VectorType_t<VBase<T>, 2>;
        using Base = typename VectorTraits<T>::base;
        using InstanceType = BinaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<T>, "Saturate only works with non cuda vector types");
            return Max<Base>::exec(params.x, Min<Base>::exec(input, params.y));
        }
    };

    template <typename T>
    struct SaturateFloat {
        UNARY_DECL_EXEC(T, T) {
            static_assert(std::is_same_v<VBase<T>, float>, "Satureate float only works with float base types.");
            return UnaryV<T,T,SaturateFloatBase>::exec(input);
        }
    };
} // namespace fk
