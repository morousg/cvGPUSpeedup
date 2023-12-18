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

#include <fused_kernel/core/utils/type_lists.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.cuh>
#include <cuda_runtime.h>

namespace fk {
    template <int SIZE>
    struct TypeSize {};

    template <>
    struct TypeSize<1> {
        enum { size = 1 };
        using default_type = uchar;
    };
    template <>
    struct TypeSize<2> {
        enum { size = 2 };
        using default_type = ushort;
    };
    template <>
    struct TypeSize<4> {
        enum { size = 4 };
        using default_type = uint;
    };
    template <>
    struct TypeSize<8> {
        enum { size = 8 };
        using default_type = ulonglong;
    };

    template <typename OriginalType, int times_bigger, typename Enabler=void>
    struct BiggerTypeSelect {};

    template <typename OriginalType, int times_bigger>
    struct BiggerTypeSelect<OriginalType, times_bigger, std::enable_if_t<std::is_aggregate_v<OriginalType>>> {
        using type = VectorType_t<typename TypeSize<sizeof(OriginalType)>::default_type, times_bigger>;
    };

    template <typename OriginalType, int times_bigger>
    struct BiggerTypeSelect<OriginalType, times_bigger, std::enable_if_t<!std::is_aggregate_v<OriginalType>>> {
        using type = VectorType_t<OriginalType, times_bigger>;
    };

    template <typename OriginalType, int times_bigger>
    using BiggerType = typename BiggerTypeSelect<OriginalType, times_bigger>::type;

    template <typename OriginalType, int TIMES_BIGGER>
    struct ThreadFusionInfo {};

    template <typename OriginalType>
    struct ThreadFusionInfo<OriginalType, 1> {
        enum { times_bigger = 1 };
        using type = OriginalType;
    };

    template <typename OriginalType>
    struct ThreadFusionInfo<OriginalType, 2> {
        enum { times_bigger = 2 };
        using type = BiggerType<OriginalType, 2>;
        template <int IDX>
        FK_HOST_DEVICE_FUSE OriginalType get(const type& data) {
            if constexpr (std::is_aggregate_v<OriginalType>) {
                if constexpr (IDX == 0) {
                    const OriginalType* const smallerData = (OriginalType*)&(data.x);
                    return *smallerData;
                } else {
                    const OriginalType* const smallerData = (OriginalType*)&(data.y);
                    return *smallerData;
                }
            } else {
                if constexpr (IDX == 0) {
                    return data.x;
                } else {
                    return data.y;
                }
            }
        }
    };

    template <typename OriginalType>
    struct ThreadFusionInfo<OriginalType, 4> {
        enum { times_bigger = 4 };
        using type = BiggerType<OriginalType, 4>;
        template <int IDX>
        FK_HOST_DEVICE_FUSE OriginalType get(const type& data) {
            static_assert(IDX <= times_bigger, "The BiggerType has not so many elements.");
            if constexpr (std::is_aggregate_v<OriginalType>) {
                if constexpr (IDX == 0) {
                    const OriginalType* const smallerData = (OriginalType*)&(data.x);
                    return *smallerData;
                } else if constexpr (IDX == 1) {
                    const OriginalType* const smallerData = (OriginalType*)&(data.y);
                    return *smallerData;
                } else if constexpr (IDX == 2) {
                    const OriginalType* const smallerData = (OriginalType*)&(data.z);
                    return *smallerData;
                } else {
                    const OriginalType* const smallerData = (OriginalType*)&(data.w);
                    return *smallerData;
                }
            } else {
                if constexpr (IDX == 0) {
                    return data.x;
                } else if constexpr (IDX == 1) {
                    return data.y;
                } else if constexpr (IDX == 2) {
                    return data.z;
                } else {
                    return data.w;
                }
            }
        }
    };

    using TypeSizes =
        TypeList<TypeSize<1>, TypeSize<2>,
                 TypeSize<3>, TypeSize<4>,
                 TypeSize<6>, TypeSize<8>,
                 TypeSize<12>, TypeSize<16>,
                 TypeSize<24>, TypeSize<32>>;

    template <typename OriginalType>
    using ThreadFusionInfos =
        TypeList<ThreadFusionInfo<OriginalType, 4>, ThreadFusionInfo<OriginalType, 4>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 4>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 2>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 1>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 1>>;

    template <typename OriginalType>
    using ThreadFusionInfo_t = EquivalentType_t<TypeSize<sizeof(OriginalType)>, TypeSizes, ThreadFusionInfos<OriginalType>>;
} // namespace fk
