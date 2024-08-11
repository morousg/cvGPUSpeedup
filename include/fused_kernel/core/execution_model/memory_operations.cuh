/* 
   Copyright 2023-2024 Oscar Amoros Huguet

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
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>

namespace fk {

    template <ND D, typename T>
    struct PerThreadRead {
        using ParamsType = RawPtr<D, T>;
        using ReadDataType = T;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;

        template <uint ELEMS_PER_THREAD=1>
        FK_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& ptr) {
            return *PtrAccessor<D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, ptr);
        }

        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <ND D, typename T>
    struct PerThreadWrite {
        using ParamsType = RawPtr<D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<T, ELEMS_PER_THREAD>& input, const ParamsType& output) {
            *PtrAccessor<D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, output) = input;
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorRead {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;
        using ReadDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& ptr) {
            return *PtrAccessor<_3D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, ptr);
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorWrite {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<T, ELEMS_PER_THREAD>& input, const ParamsType& output) {
            *PtrAccessor<_3D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, output) = input;
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorSplit {
        using WriteDataType = VBase<T>;
        using ParamsType = RawPtr<_3D, WriteDataType>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;

        FK_DEVICE_FUSE void exec(const Point& thread, const T& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = ptr.dims.width * ptr.dims.height;

            WriteDataType* const work_plane = PtrAccessor<_3D>::point(thread, ptr);

            *work_plane = input.x;
            *(work_plane + planePixels) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *(work_plane + (planePixels * 2)) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *(work_plane + (planePixels * 3)) = input.w;
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorTSplit {
        using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;
        using WriteDataType = VBase<InputType>;

        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            *PtrAccessor<T3D>::point(thread, ptr, 0) = input.x;
            *PtrAccessor<T3D>::point(thread, ptr, 1) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *PtrAccessor<T3D>::point(thread, ptr, 2) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *PtrAccessor<T3D>::point(thread, ptr, 3) = input.w;
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorPack {
        using ParamsType = RawPtr<_3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        FK_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = ptr.dims.width * ptr.dims.height;

            const VBase<T>* const work_plane = PtrAccessor<_3D>::cr_point(thread, ptr);
            if constexpr (cn<OutputType> == 2) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels));
            } else if constexpr (cn<OutputType> == 3) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)));
            } else {
                return make_<OutputType>(*work_plane,
                    *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)),
                    *(work_plane + (planePixels * 3)));
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <typename T>
    struct TensorTPack {
        using ParamsType = RawPtr<T3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        FK_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const VBase<T> x = *PtrAccessor<T3D>::cr_point(thread, ptr, 0);
            if constexpr (cn<OutputType> == 2) {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                return make_<OutputType>(x, y);
            } else if constexpr (cn<OutputType> == 3) {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                const VBase<T> z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
                return make_<OutputType>(x, y, z);
            } else {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                const VBase<T> z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
                const VBase<T> w = *PtrAccessor<T3D>::cr_point(thread, ptr, 3);
                return make_<OutputType>(x, y, z, w);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
    };

    template <ND D, typename T, typename Enabler = void>
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
        using ParamsType = SplitWriteParams<D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;
        using WriteDataType = VBase<T>;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2, "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, params.x) = input.x;
            *PtrAccessor<D>::point(thread, params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.x.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.x.dims.pitch;
        }
    };

    template <typename Operation, int NPtr, typename Enabler=void>
    struct BatchRead {};

    template <typename Operation, int NPtr>
    struct BatchRead<Operation, NPtr, std::enable_if_t<isReadBackType<Operation>>> {
        using InstanceType = ReadBackType;
        using ParamsType = typename Operation::ParamsType[NPtr];
        using BackFunction = typename Operation::BackFunction[NPtr];
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ false };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE const auto exec(const Point& thread, const typename Operation::ParamsType(&params)[NPtr],
            const typename Operation::BackFunction(&back_function)[NPtr]) {
            return Operation::exec(thread, params[thread.z], back_function[thread.z]);
        }
    };

    template <typename Operation, int NPtr>
    struct BatchRead<Operation, NPtr, std::enable_if_t<isReadType<Operation>>> {
        using InstanceType = ReadType;
        using ParamsType = typename Operation::ParamsType[NPtr];
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD=1>
        FK_DEVICE_FUSE const auto exec(const Point& thread, const typename Operation::ParamsType(&params)[NPtr]) {
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(thread, params[thread.z]);
            } else {
                return Operation::exec(thread, params[thread.z]);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr[thread.z]);
        }
    };

    template <typename Operation, int NPtr>
    struct BatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = typename Operation::ParamsType[NPtr];
        using InstaceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const typename Operation::ParamsType(&params)[NPtr]) {
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(thread, input, params[thread.z]);
            } else {
                Operation::exec(thread, input, params[thread.z]);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr[thread.z]);
        }
    };

    /* The following code has the following copy right

       Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huget)
       Copyright 2023 Mediaproduccion S.L.U. (Guillermo Oyarzun Altamirano)

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License. */

    enum CircularDirection { Ascendent, Descendent };

    template <typename ParamsType>
    struct CircularMemoryParams {
        uint first;
        ParamsType params;
    };

    template <CircularDirection direction, int BATCH>
    FK_HOST_DEVICE_CNST Point computeCircularThreadIdx(const Point& currentIdx, const uint& fst) {
        if constexpr (direction == CircularDirection::Ascendent) {
            const uint z = currentIdx.z + fst;
            return { currentIdx.x, currentIdx.y, z >= BATCH ? z - BATCH : z };
        } else {
            const int z = fst - currentIdx.z;
            return { currentIdx.x, currentIdx.y, z < 0 ? static_cast<uint>(BATCH + z) : static_cast<uint>(z) };
        }
    }

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchRead {
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, c_params.params[newThreadIdx.z]);
            } else {
                return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params[thread.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, c_params.params[newThreadIdx.z]);
            } else {
                Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params[thread.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, c_params.params);
            } else {
                return Operation::exec(newThreadIdx, c_params.params);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, c_params.params);
            } else {
                Operation::exec(newThreadIdx, input, c_params.params);
            }
        }
        FK_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params);
        }
    };

    enum ROI { OFFSET_THREADS, KEEP_THREAD_IDX };

    template <typename T>
    struct ApplyROIParams {
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <typename BackFunction_, ROI USE>
    struct ApplyROI {
        using BackFunction = BackFunction_;
        using OutputType = GetOutputType_t<BackFunction>;
        using ParamsType = ApplyROIParams<OutputType>;
        using ReadDataType = OutputType;
        using InstanceType = ReadBackType;
        static constexpr bool THREAD_FUSION{ false };

        static const __device__ __forceinline__ OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            if (thread.x >= params.x1 && thread.x <= params.x2 && thread.y >= params.y1 && thread.y <= params.y2) {
                if constexpr (USE == OFFSET_THREADS) {
                    const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                    return read(roiThread, back_function);
                } else {
                    return read(thread, back_function);
                }
            } else {
                return params.defaultValue;
            }
        }
    };

} //namespace fk
