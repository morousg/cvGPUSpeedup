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
    struct TypeSize : std::false_type {};

    template <>
    struct TypeSize<1> : std::true_type {
        enum { size = 1 };
        using default_type = uchar;
    };

    template <>
    struct TypeSize<2> : std::true_type {
        enum { size = 2 };
        using default_type = ushort;
    };

    template <>
    struct TypeSize<4> : std::true_type {
        enum { size = 4 };
        using default_type = uint;
    };

    template <>
    struct TypeSize<8> : std::true_type {
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
        static constexpr bool ENABLED{ TypeSize<sizeof(OriginalType)>::value };
        enum { times_bigger = 1 };
        using type = OriginalType;
    };

    template <typename OriginalType>
    struct ThreadFusionInfo<OriginalType, 2> {
        static constexpr bool ENABLED{ true };
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
        FK_HOST_DEVICE_FUSE type make(const OriginalType& data0, const OriginalType& data1) {
            if constexpr (std::is_same_v<VBase<type>, OriginalType>) {
                return make_<type>(data0, data1);
            } else {
                const VBase<type> tempO0_ = *((VBase<type>*) & data0);
                const VBase<type> tempO1_ = *((VBase<type>*) & data1);
                return make_<type>(tempO0_, tempO1_);
            }
        }
    };

    template <typename OriginalType>
    struct ThreadFusionInfo<OriginalType, 4> {
        static constexpr bool ENABLED{ true };
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
        FK_HOST_DEVICE_FUSE type make(const OriginalType& data0, const OriginalType& data1,
                                      const OriginalType& data2, const OriginalType& data3) {
            if constexpr (std::is_same_v<VBase<type>, OriginalType>) {
                return make_<type>(data0, data1, data2,  data3);
            } else {
                const VBase<type> tempO0_ = *((VBase<type>*) & data0);
                const VBase<type> tempO1_ = *((VBase<type>*) & data1);
                const VBase<type> tempO2_ = *((VBase<type>*) & data2);
                const VBase<type> tempO3_ = *((VBase<type>*) & data3);
                return make_<type>(tempO0_, tempO1_, tempO2_, tempO3_);
            }
        }
    };

    using TypeSizes =
                 TypeList<TypeSize<1>,              TypeSize<2>,
                 TypeSize<3>,                       TypeSize<4>,
                 TypeSize<6>,                       TypeSize<8>,
                 TypeSize<12>,                      TypeSize<16>,
                 TypeSize<24>,                      TypeSize<32>>;

    template <typename OriginalType>
    using ThreadFusionInfos =
        TypeList<ThreadFusionInfo<OriginalType, 4>, ThreadFusionInfo<OriginalType, 4>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 4>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 2>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 1>,
                 ThreadFusionInfo<OriginalType, 1>, ThreadFusionInfo<OriginalType, 1>>;

    template <typename OriginalType>
    using ThreadFusionInfo_t = EquivalentType_t<TypeSize<sizeof(OriginalType)>, TypeSizes, ThreadFusionInfos<OriginalType>>;

    template <typename... ThreadFusionInfos>
    constexpr bool allTFEnabled = (ThreadFusionInfos::ENABLED && ...);
} // namespace fk
