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

#ifndef FK_MEMORY_OPERATIONS
#define FK_MEMORY_OPERATIONS

#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {

    template <ND D, typename T>
    struct PerThreadRead {
        using ParamsType = RawPtr<D, T>;
        using ReadDataType = T;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;

        template <uint ELEMS_PER_THREAD=1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& ptr) {
            return *PtrAccessor<D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, ptr);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Read<PerThreadRead<D, T>>;
        DEFAULT_READ_BUILD
    };

    template <ND D, typename T>
    struct PerThreadWrite {
        using ParamsType = RawPtr<D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<T, ELEMS_PER_THREAD>& input, const ParamsType& output) {
            *PtrAccessor<D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, output) = input;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Write<PerThreadWrite<D, T>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename T>
    struct TensorRead {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;
        using ReadDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& ptr) {
            return *PtrAccessor<_3D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, ptr);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Read<TensorRead<T>>;
        DEFAULT_READ_BUILD
    };

    template <typename T>
    struct TensorWrite {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<T, ELEMS_PER_THREAD>& input, const ParamsType& output) {
            *PtrAccessor<_3D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, output) = input;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Write<TensorWrite<T>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename T>
    struct TensorSplit {
        using WriteDataType = VBase<T>;
        using ParamsType = RawPtr<_3D, WriteDataType>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;

        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const T& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Write<TensorSplit<T>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename T>
    struct TensorTSplit {
        using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;
        using WriteDataType = VBase<InputType>;

        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            *PtrAccessor<T3D>::point(thread, ptr, 0) = input.x;
            *PtrAccessor<T3D>::point(thread, ptr, 1) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *PtrAccessor<T3D>::point(thread, ptr, 2) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *PtrAccessor<T3D>::point(thread, ptr, 3) = input.w;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Write<TensorTSplit<T>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename T>
    struct TensorPack {
        using ParamsType = RawPtr<_3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Read<TensorPack<T>>;
        DEFAULT_READ_BUILD
    };

    template <typename T>
    struct TensorTPack {
        using ParamsType = RawPtr<T3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Read<TensorTPack<T>>;
        DEFAULT_READ_BUILD
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
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, params.x) = input.x;
            *PtrAccessor<D>::point(thread, params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return ptr.x.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.x.dims.pitch;
        }
        using InstantiableType = Write<SplitWrite<D, T>>;
        DEFAULT_WRITE_BUILD
    };

    template <int BATCH, typename Operation = void>
    struct BatchReadBack {
        using InstanceType = ReadBackType;
        using ParamsType = typename Operation::ParamsType[BATCH];
        using BackFunction = typename Operation::BackFunction[BATCH];
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ false };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const typename Operation::ParamsType(&params)[BATCH],
            const typename Operation::BackFunction(&back_function)[BATCH]) {
            return Operation::exec(thread, params[thread.z], back_function[thread.z]);
        }
        using InstantiableType = ReadBack<BatchReadBack<BATCH, Operation>>;
        static constexpr __host__  __forceinline__ 
        InstantiableType build(const ParamsType& params, const BackFunction& backFunction) {
            return InstantiableType{ { params, backFunction } };
        }

        using InstantiableSourceType = SourceReadBack<BatchReadBack<BATCH, Operation>>;
    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<ReadBack<Operation>, BATCH>& deviceFunctions,
                                                   const std::integer_sequence<int, Idx...>&) {
            return { {{deviceFunctions[Idx].params...}, {deviceFunctions[Idx].back_function...}} };
        }

        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                                                   const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                                                   const std::integer_sequence<int, Idx...>&) {
            return { {{params[Idx]...}, {back_functions[Idx]...}} };
        }

        template <int... Idx>
        FK_HOST_FUSE InstantiableSourceType build_source_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                                                                const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                                                                const Size& output_planes, const std::integer_sequence<int, Idx...>&) {
            return { {{params[Idx]...}, {back_functions[Idx]...}},
            {static_cast<uint>(output_planes.width), static_cast<uint>(output_planes.height), static_cast<uint>(BATCH)} };
        }
    public:
        FK_HOST_FUSE InstantiableType build(const std::array<ReadBack<Operation>, BATCH>& deviceFunctions) {
            return build_helper(deviceFunctions, std::make_integer_sequence<int, BATCH>{});
        }

        FK_HOST_FUSE InstantiableType build(const std::array<typename Operation::ParamsType, BATCH>& params,
                                            const std::array<typename Operation::BackFunction, BATCH>& back_functions) {
            return build_helper(params, back_functions, std::make_integer_sequence<int, BATCH>{});
        }

        FK_HOST_FUSE InstantiableSourceType build_source(const std::array<typename Operation::ParamsType, BATCH>& params,
                                                         const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                                                         const Size& output_planes) {
            return build_source_helper(params, back_functions, output_planes, std::make_integer_sequence<int, BATCH>{});
        }
    };

    template <int BATCH>
    struct BatchReadBack<BATCH, void> {
    private:
        // Base case: Only one Operation
        template <typename Operation, typename ParamsArray>
        FK_HOST_FUSE auto nested_build_ops(const size_t& idx, const ParamsArray& params_array) {
            return Operation::build(params_array[idx]);
        }

        // Recursive case: Multiple Operations
        template <typename Operation, typename... RestOperations, typename ParamsArray, typename... RestParamsArrays>
        FK_HOST_FUSE auto nested_build_ops(const size_t& idx, const ParamsArray& params_array, const RestParamsArrays&... rest_params_arrays) {
            return Operation::build(params_array[idx], nested_build_ops<RestOperations...>(idx, rest_params_arrays...));
        }

        template <typename... Operations, size_t... Idx, typename... ParamsArrays>
        FK_HOST_FUSE auto nested_build_ops(const std::index_sequence<Idx...>&, const ParamsArrays&... params_arrays) {
            static_assert(sizeof...(Operations) == sizeof...(ParamsArrays), "Operations and ParamsArrays should have the same size");
            using OperationsArraysTypes = TypeList<TypeList<Operations, ParamsArrays>...>;
            return std::array<BackType<OperationsArraysTypes>, sizeof...(Idx)>{nested_build_ops<Operations...>(Idx, params_arrays...)...};
        }

        template <typename... Operations, typename... ParamsArrays>
        FK_HOST_FUSE auto build_batch_back(const ParamsArrays&... params_arrays) {
            static_assert(sizeof...(Operations) == sizeof...(ParamsArrays), "Operations and ParamsArrays should have the same size");
            return nested_build_ops<Operations...>(std::make_index_sequence<BATCH>{}, params_arrays...);
        }
    public:
        template <typename Operation>
        FK_HOST_FUSE auto build(const std::array<ReadBack<Operation>, BATCH>& deviceFunctions) {
            return BatchReadBack<BATCH, Operation>::build(deviceFunctions);
        }

        template <typename... Operations, typename... ParamsArrays>
        FK_HOST_FUSE auto build(const ParamsArrays&... params_arrays) {
            return BatchReadBack<BATCH, void>::build(build_batch_back<Operations...>(params_arrays...));
        }
    };

    template <int BATCH, typename Operation>
    struct BatchRead {
        using InstanceType = ReadType;
        using ParamsType = typename Operation::ParamsType[BATCH];
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD=1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const typename Operation::ParamsType(&params)[BATCH]) {
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(thread, params[thread.z]);
            } else {
                return Operation::exec(thread, params[thread.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr[thread.z]);
        }
        using InstantiableType = Read<BatchRead<BATCH, Operation>>;
        DEFAULT_READ_BUILD

        using InstantiableSourceType = SourceRead<BatchRead<BATCH, Operation>>;
    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableSourceType build_source_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                                                                const Size& output_planes, const std::integer_sequence<int, Idx...>&) {
            return { {{params[Idx]...}},
            {static_cast<uint>(output_planes.width), static_cast<uint>(output_planes.height), static_cast<uint>(BATCH)} };
        }
    public:
        FK_HOST_FUSE InstantiableSourceType build_source(const std::array<typename Operation::ParamsType, BATCH>& params,
                                                         const Size& output_planes) {
            return build_source_helper(params, output_planes, std::make_integer_sequence<int, BATCH>{});
        }
    };

    template <int BATCH, typename Operation>
    struct BatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = typename Operation::ParamsType[BATCH];
        using InstaceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input,
                                      const typename Operation::ParamsType(&params)[BATCH]) {
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(thread, input, params[thread.z]);
            } else {
                Operation::exec(thread, input, params[thread.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr[thread.z]);
        }
        using InstantiableType = Write<BatchWrite<BATCH, Operation>>;
        DEFAULT_WRITE_BUILD
    };

    /* The following code has the following copy right

       Copyright 2024 Oscar Amoros Huguet
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
        FK_HOST_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, c_params.params[newThreadIdx.z]);
            } else {
                return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params[thread.z]);
        }
        using InstantiableType = Read<CircularBatchRead<direction, Operation, BATCH>>;
        DEFAULT_READ_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, c_params.params[newThreadIdx.z]);
            } else {
                Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params[thread.z]);
        }
        using InstantiableType = Write<CircularBatchWrite<direction, Operation, BATCH>>;
        DEFAULT_WRITE_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, c_params.params);
            } else {
                return Operation::exec(newThreadIdx, c_params.params);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params);
        }
        using InstantiableType = Read<CircularTensorRead<direction, Operation, BATCH>>;
        DEFAULT_READ_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, c_params.params);
            } else {
                Operation::exec(newThreadIdx, input, c_params.params);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.params);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.params);
        }
        using InstantiableType = Write<CircularTensorWrite<direction, Operation, BATCH>>;
        DEFAULT_WRITE_BUILD
    };

    enum ROI { OFFSET_THREADS, KEEP_THREAD_IDX };

    template <typename T>
    struct ApplyROIParams {
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <ROI USE, typename BackFunction_ = void>
    struct ApplyROI {
        using BackFunction = BackFunction_;
        using OutputType = GetOutputType_t<BackFunction>;
        using ParamsType = ApplyROIParams<OutputType>;
        using ReadDataType = OutputType;
        using InstanceType = ReadBackType;
        static constexpr bool THREAD_FUSION{ false };

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
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

        using InstantiableType = ReadBack<ApplyROI<USE, BackFunction_>>;
        static constexpr __host__ __forceinline__ auto build(const ParamsType& params, const BackFunction_& backFunction) {
            return InstantiableType{ {params, backFunction} };
        }
    };

    template <ROI USE>
    struct ApplyROI<USE, void> {
        template <typename RealBackFuntion>
        static constexpr __host__ __forceinline__
            auto build(const typename ApplyROI<USE, RealBackFuntion>::ParamsType& params, const RealBackFuntion& backFunction) {
            return ApplyROI<USE, RealBackFuntion>::build(params, backFunction);
        }
    };

} //namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
