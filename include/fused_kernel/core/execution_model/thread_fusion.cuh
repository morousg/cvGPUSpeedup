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
#include <fused_kernel/core/utils/template_operations.cuh>
#include <fused_kernel/core/data/vector_types.cuh>
#include <cuda_runtime.h>

namespace fk {

    /* Possible combinations:
    Size, Channels, Types
    1,    1         char, uchar                     8,  8  char4, uchar4        x4
    2,    1         short, ushort                   4,  2  short2, ushort2      x2
    4,    1         int, uint, float                8,  2  int2, uint2, float2  x2
    8,    1         longlong, ulonglong, double     8,  1
    2,    2         char2, uchar2                   4,  4  char4, uchar4        x2
    4,    2         short2, ushort2                 4,  2
    8,    2         int2, uint2, float2             8,  2
    16,   2         longlong2, ulonglong2, double2  16, 2
    3,    3         char3, uchar3                   3,  3
    6,    3         short3, ushort3                 6,  3
    12,   3         int3, uint3, float3             12, 3
    24,   3         longlong3, ulonglong3, double3  24, 3
    4,    4         char4, uchar4                   4,  4
    8,    4         short4, ushort4                 8,  4
    16,   4         int4, uint4, float4             16, 4
    32,   4         longlong4, ulonglong4, double4  32, 4

    Times bigger can be: 1, 2, 4
    */

    using TFSourceTypes = typename TypeList<StandardTypes, VTwo, VThree, VFour>::type;
    using TFBiggerTypes = TypeList<uchar4,  char4,  ushort2,  short2,  uint2, int2, ulong,  long,  ulonglong,  longlong,  float2, double,
                                   uchar4,  char4,  ushort2,  short2,  uint2, int2, ulong2, long2, ulonglong2, longlong2, float2, double2,
                                   uchar3,  char3,  ushort3,  short3,  uint3, int3, ulong3, long3, ulonglong3, longlong3, float3, double3,
                                   uchar4,  char4,  ushort4,  short4,  uint4, int4, ulong4, long4, ulonglong4, longlong4, float4, double4>;
    template <typename SourceType>
    using TFBiggerType_t = EquivalentType_t<SourceType, TFSourceTypes, TFBiggerTypes>;

    constexpr std::integer_sequence<uint, 1, 2, 3, 4> validChannelsSequence;

    template <uint channelNumber>
    constexpr bool isValidChannelNumber = Find<uint, channelNumber>::one_of(validChannelsSequence);

    template <typename SourceType, uint ELEMS_PER_THREAD>
    using ThreadFusionType = VectorType_t<VBase<SourceType>, cn<SourceType>* ELEMS_PER_THREAD>;

    template <typename ReadType, typename WriteType, bool ENABLED_>
    struct ThreadFusionInfo {
        public:
            static constexpr bool ENABLED = ENABLED_ && isValidChannelNumber<(cn<TFBiggerType_t<ReadType>> / cn<ReadType>) * cn<WriteType>>;
            using BiggerReadType = std::conditional_t<ENABLED, TFBiggerType_t<ReadType>, ReadType>;
            static constexpr uint elems_per_thread{ cn<BiggerReadType> / cn<ReadType> };
            using BiggerWriteType = VectorType_t<VBase<WriteType>, elems_per_thread * cn<WriteType>>;

            template <int IDX>
            FK_HOST_DEVICE_FUSE ReadType get(const BiggerReadType& data) {
                static_assert(IDX < elems_per_thread, "Index out of range for this ThreadFusionInfo");
                if constexpr (cn<ReadType> == 1) {
                    if constexpr (IDX == 0) {
                        return data.x;
                    } else if constexpr (IDX == 1) {
                        return data.y;
                    } else if constexpr (IDX == 2) {
                        return data.z;
                    } else if constexpr (IDX == 3) {
                        return data.w;
                    }
                } else if constexpr (cn<ReadType> == 2) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y);
                    } else if constexpr (IDX == 1) {
                        return make_<ReadType>(data.z, data.w);
                    }
                } else if constexpr (cn<ReadType> == 3) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y, data.z);
                    }
                } else if constexpr (cn<ReadType> == 4) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y, data.z, data.w);
                    }
                }
            }
            template <typename... OriginalTypes>
            FK_HOST_DEVICE_FUSE BiggerWriteType make(const OriginalTypes&... data) {
                static_assert(and_v<std::is_same_v<WriteType, OriginalTypes>...>, "Not all types are the same when making the ThreadFusion BiggerType value");
                if constexpr (cn<WriteType> > 1) {
                    return make_impl(data...);
                } else {
                    return make_<BiggerWriteType>(data...);
                }
            }

        private:
            FK_HOST_DEVICE_FUSE BiggerWriteType make_impl(const WriteType& data0,
                                                          const WriteType& data1) {
                if constexpr (cn<WriteType> == 2) {
                    return make_<BiggerWriteType>(data0.x, data0.y, data1.x, data1.y);
                }
            }
    };

    // Thread Fusion hepler functions

    template <bool THREAD_FUSION_ENABLED, typename... DeviceFunctionTypes>
    inline constexpr bool isThreadFusionEnabled() {
        using ReadOperation = typename FirstType_t<DeviceFunctionTypes...>::Operation;
        using WriteOperation = typename LastType_t<DeviceFunctionTypes...>::Operation;
        return ReadOperation::THREAD_FUSION && WriteOperation::THREAD_FUSION && THREAD_FUSION_ENABLED;
    }

    template <bool THREAD_FUSION_ENABLED, typename... DeviceFunctionTypes>
    inline constexpr uint computeElementsPerThread(const DeviceFunctionTypes&... deviceFunctions) {
        using ReadOperation = typename FirstType_t<DeviceFunctionTypes...>::Operation;
        using WriteOperation = typename LastType_t<DeviceFunctionTypes...>::Operation;
        using RDT = typename ReadOperation::ReadDataType;
        using WDT = typename WriteOperation::WriteDataType;
        if constexpr (THREAD_FUSION_ENABLED) {
            using TFI = ThreadFusionInfo<RDT, WDT, THREAD_FUSION_ENABLED>;
            const bool pitch_divisible =
                (ReadOperation::pitch(Point(0, 0, 0), first(deviceFunctions...).params) % (sizeof(RDT) * TFI::elems_per_thread) == 0) &&
                (WriteOperation::pitch(Point(0, 0, 0), last(deviceFunctions...).params) % (sizeof(WDT) * TFI::elems_per_thread) == 0);
            return pitch_divisible ? TFI::elems_per_thread : 1u;
        } else {
            return 1u;
        }
    }

    template <bool THREAD_FUSION_ENABLED, typename... DeviceFunctionTypes>
    inline bool isThreadDivisible(const uint& elems_per_thread, const DeviceFunctionTypes&... deviceFunctions) {
        if constexpr (THREAD_FUSION_ENABLED) {
            const auto& readOp = first(deviceFunctions...);
            const auto& writeOp = last(deviceFunctions...);
            using ReadOperation = typename FirstType_t<DeviceFunctionTypes...>::Operation;
            using WriteOperation = typename LastType_t<DeviceFunctionTypes...>::Operation;
            const uint readRow = ReadOperation::num_elems_x(Point(0, 0, 0), readOp.params);
            const uint writeRow = WriteOperation::num_elems_x(Point(0, 0, 0), writeOp.params);
            return (readRow % elems_per_thread == 0) && (writeRow % elems_per_thread == 0);
        } else {
            return true;
        }
    }

} // namespace fk
