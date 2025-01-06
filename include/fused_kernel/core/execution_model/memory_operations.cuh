/* Copyright 2023-2025 Oscar Amoros Huguet

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
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>
#include <fused_kernel/core/data/array.cuh>
#include <vector>

namespace fk {

    template <typename InstantiableOp, typename Enabler = void>
    struct Num_elems;

    template <typename InstantiableOp>
    struct Num_elems<InstantiableOp, std::enable_if_t<InstantiableOp::template is<ReadType>, void>> {
        FK_HOST_DEVICE_FUSE uint x(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_x(thread, iOp);
        }
        FK_HOST_DEVICE_FUSE uint y(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_y(thread, iOp);
        }
        FK_HOST_DEVICE_FUSE Size size(const Point& thread, const InstantiableOp& iOp) {
            return Size(x(thread, iOp), y(thread, iOp));
        }
        FK_HOST_DEVICE_FUSE uint z(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_z(thread, iOp);
        }
    };

    template <typename InstantiableOp>
    struct Num_elems<InstantiableOp, std::enable_if_t<InstantiableOp::template is<ReadBackType>, void>> {
        FK_HOST_DEVICE_FUSE uint x(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_x(thread, iOp);
        }

        FK_HOST_DEVICE_FUSE uint y(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_y(thread, iOp);
        }
        FK_HOST_DEVICE_FUSE Size size(const Point& thread, const InstantiableOp& iOp) {
            return Size(x(thread, iOp), y(thread, iOp));
        }
        FK_HOST_DEVICE_FUSE uint z(const Point& thread, const InstantiableOp& iOp) {
            return InstantiableOp::Operation::num_elems_z(thread, iOp);
        }
    };

    template <enum ND D, typename T>
    struct PerThreadRead {
        using ParamsType = RawPtr<D, T>;
        using ReadDataType = T;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;
        using OperationDataType = OperationData<PerThreadRead<D, T>>;
        template <uint ELEMS_PER_THREAD=1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD>
        exec(const Point& thread, const OperationDataType& opData) {
            return *PtrAccessor<D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, opData.params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            if constexpr (D == _1D) {
                return 1;
            } else {
                return opData.params.dims.height;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            if constexpr (D == _1D || D == _2D) {
                return 1;
            } else {
                return opData.params.dims.planes;
            }
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Read<PerThreadRead<D, T>>;
        DEFAULT_BUILD
        DEFAULT_BUILD_PARAMS
        DEFAULT_READ_BATCH_BUILD
    };

    template <enum ND D, typename T>
    struct PerThreadWrite {
        using ParamsType = RawPtr<D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;
        using OperationDataType = OperationData<PerThreadWrite<D, T>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<T, ELEMS_PER_THREAD>& input,
                                      const OperationDataType& opData) {
            *PtrAccessor<D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, opData.params) = input;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        using InstantiableType = Write<PerThreadWrite<D, T>>;
        DEFAULT_BUILD
    };

    template <typename T>
    struct TensorRead {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ true };
        using OutputType = T;
        using ReadDataType = T;
        using OperationDataType = OperationData<TensorRead<T>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<T, ELEMS_PER_THREAD> exec(const Point& thread, const OperationDataType& opData) {
            return *PtrAccessor<_3D>::template cr_point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, opData.params);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Read<TensorRead<T>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename T>
    struct TensorWrite {
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ true };
        using InputType = T;
        using WriteDataType = T;
        using OperationDataType = OperationData<TensorWrite<T>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<T, ELEMS_PER_THREAD>& input, const OperationDataType& opData) {
            *PtrAccessor<_3D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD>>(thread, opData.params) = input;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        using InstantiableType = Write<TensorWrite<T>>;
        DEFAULT_BUILD
    };

    template <typename T>
    struct TensorSplit {
        using WriteDataType = VBase<T>;
        using ParamsType = RawPtr<_3D, WriteDataType>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;
        using OperationDataType = OperationData<TensorSplit<T>>;

        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const T& input, const OperationDataType& opData) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = opData.params.dims.width * opData.params.dims.height;

            WriteDataType* const work_plane = PtrAccessor<_3D>::point(thread, opData.params);

            *work_plane = input.x;
            *(work_plane + planePixels) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *(work_plane + (planePixels * 2)) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *(work_plane + (planePixels * 3)) = input.w;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        using InstantiableType = Write<TensorSplit<T>>;
        DEFAULT_BUILD
    };

    template <typename T>
    struct TensorTSplit {
        using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ false };
        using InputType = T;
        using WriteDataType = VBase<InputType>;
        using OperationDataType = OperationData<TensorTSplit<T>>;

        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const OperationDataType& opData) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            *PtrAccessor<T3D>::point(thread, opData.params, 0) = input.x;
            *PtrAccessor<T3D>::point(thread, opData.params, 1) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *PtrAccessor<T3D>::point(thread, opData.params, 2) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *PtrAccessor<T3D>::point(thread, opData.params, 3) = input.w;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        using InstantiableType = Write<TensorTSplit<T>>;
        DEFAULT_BUILD
    };

    template <typename T>
    struct TensorPack {
        using ParamsType = RawPtr<_3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        using OperationDataType = OperationData<TensorPack<T>>;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = opData.params.dims.width * opData.params.dims.height;

            const VBase<T>* const work_plane = PtrAccessor<_3D>::cr_point(thread, opData.params);
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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
        using InstantiableType = Read<TensorPack<T>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename T>
    struct TensorTPack {
        using ParamsType = RawPtr<T3D, VBase<T>>;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        using OutputType = T;
        using ReadDataType = T;
        using OperationDataType = OperationData<TensorTPack<T>>;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const OperationDataType& opData) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const VBase<T> x = *PtrAccessor<T3D>::cr_point(thread, opData.params, 0);
            if constexpr (cn<OutputType> == 2) {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, opData.params, 1);
                return make_<OutputType>(x, y);
            } else if constexpr (cn<OutputType> == 3) {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, opData.params, 1);
                const VBase<T> z = *PtrAccessor<T3D>::cr_point(thread, opData.params, 2);
                return make_<OutputType>(x, y, z);
            } else {
                const VBase<T> y = *PtrAccessor<T3D>::cr_point(thread, opData.params, 1);
                const VBase<T> z = *PtrAccessor<T3D>::cr_point(thread, opData.params, 2);
                const VBase<T> w = *PtrAccessor<T3D>::cr_point(thread, opData.params, 3);
                return make_<OutputType>(x, y, z, w);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
        using InstantiableType = Read<TensorTPack<T>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
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
        using OperationDataType = OperationData<SplitWrite<D,T>>;

        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const OperationDataType& opData) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, opData.params.x) = input.x;
            *PtrAccessor<D>::point(thread, opData.params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, opData.params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, opData.params.w) = input.w;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.x.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.x.dims.pitch;
        }
        using InstantiableType = Write<SplitWrite<D, T>>;
        DEFAULT_BUILD
        FK_HOST_FUSE auto build(const std::vector<Ptr2D<VBase<T>>>& output) {
            static_assert(cn<T> >= 2, "Split operations can only be used with types of 2, 3 or 4 channels.");
            if constexpr (cn<T> == 2) {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr()}} };
            } else if constexpr (cn<T> == 3) {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr()}} };
            } else {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr(), output.at(3).ptr()}} };
            }
        }
    };

    template <int BATCH, typename Operation>
    struct BatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = typename Operation::ParamsType[BATCH];
        using InstaceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;
        using OperationDataType = OperationData<BatchWrite<BATCH, Operation>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input,
                                      const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(thread, input, opData.params[thread.z]);
            } else {
                Operation::exec(thread, input, opData.params[thread.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params[thread.z]);
        }
        using InstantiableType = Write<BatchWrite<BATCH, Operation>>;
        DEFAULT_BUILD
    };



    /* The following code has the following copy right

       Copyright 2024-2025 Oscar Amoros Huguet
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

    template <typename OperationDataTypeArray>
    struct CircularMemoryParams {
        uint first;
        OperationDataTypeArray opData;
    };

    namespace circular_batch_internal {
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
    } // namespace circular_batch_internal

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchRead {
        using ParamsType = CircularMemoryParams<OperationData<Operation>[BATCH]>;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;
        using OperationDataType = OperationData<CircularBatchRead<direction, Operation, BATCH>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const OperationDataType& opData) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, opData.params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, opData.params.opData[newThreadIdx.z]);
            } else {
                return Operation::exec(newThreadIdx, opData.params.opData[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Read<CircularBatchRead<direction, Operation, BATCH>>;
        DEFAULT_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<OperationData<Operation>[BATCH]>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        using OperationDataType = OperationData<CircularBatchWrite<direction, Operation, BATCH>>;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input, const OperationDataType& opBatch) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, opBatch.params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, opBatch.params.opData[newThreadIdx.z]);
            } else {
                Operation::exec(newThreadIdx, input, opBatch.params.opData[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opBatch) {
            return Operation::num_elems_x(thread, opBatch.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opBatch) {
            return Operation::pitch(thread, opBatch.params.opData[thread.z]);
        }
        using InstantiableType = Write<CircularBatchWrite<direction, Operation, BATCH>>;
        DEFAULT_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
        using ParamsType = CircularMemoryParams<OperationData<Operation>>;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using ReadDataType = typename Operation::ReadDataType;

        using OperationDataType = OperationData<CircularTensorRead<direction, Operation, BATCH>>;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD> exec(const Point& thread, const OperationDataType& opData) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, opData.params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, opData.params.opData);
            } else {
                return Operation::exec(newThreadIdx, opData.params.opData);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Read<CircularTensorRead<direction, Operation, BATCH>>;
        DEFAULT_BUILD
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<OperationData<Operation>>;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;

        using OperationDataType = OperationData<CircularTensorWrite<direction, Operation, BATCH>>;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input,
                                      const OperationDataType& opData) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, opData.params.first);
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(newThreadIdx, input, opData.params.opData);
            } else {
                Operation::exec(newThreadIdx, input, opData.params.opData);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }
        using InstantiableType = Write<CircularTensorWrite<direction, Operation, BATCH>>;
        DEFAULT_BUILD
    };

} //namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
