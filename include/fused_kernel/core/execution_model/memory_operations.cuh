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
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>
#include <fused_kernel/core/data/array.cuh>

namespace fk {

    template <typename InstantiableOp, typename Enabler = void>
    struct Num_elems;

    template <typename InstantiableOp>
    struct Num_elems<InstantiableOp, std::enable_if_t<InstantiableOp::template is<ReadType>, void>> {
        FK_HOST_DEVICE_FUSE uint x(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_x(thread, df.params);
        }
        FK_HOST_DEVICE_FUSE uint y(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_y(thread, df.params);
        }
        FK_HOST_DEVICE_FUSE Size size(const Point& thread, const InstantiableOp& df) {
            return Size(x(thread, df), y(thread, df));
        }
        FK_HOST_DEVICE_FUSE uint z(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_z(thread, df.params);
        }
    };

    template <typename InstantiableOp>
    struct Num_elems<InstantiableOp, std::enable_if_t<InstantiableOp::template is<ReadBackType>, void>> {
        FK_HOST_DEVICE_FUSE uint x(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_x(thread, df.params, df.back_function);
        }

        FK_HOST_DEVICE_FUSE uint y(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_y(thread, df.params, df.back_function);
        }
        FK_HOST_DEVICE_FUSE Size size(const Point& thread, const InstantiableOp& df) {
            return Size(x(thread, df), y(thread, df));
        }
        FK_HOST_DEVICE_FUSE uint z(const Point& thread, const InstantiableOp& df) {
            return InstantiableOp::Operation::num_elems_z(thread, df.params, df.back_function);
        }
    };

    template <enum ND D, typename T>
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

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& ptr) {
            if constexpr (D == _1D) {
                return 1;
            } else {
                return ptr.dims.height;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& ptr) {
            if constexpr (D == _1D || D == _2D) {
                return 1;
            } else {
                return ptr.dims.planes;
            }
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.pitch;
        }
        using InstantiableType = Read<PerThreadRead<D, T>>;
        DEFAULT_READ_BUILD
    };

    template <enum ND D, typename T>
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

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.planes;
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
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.planes;
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
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& ptr) {
            return ptr.dims.planes;
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

    enum PlanePolicy { PROCESS_ALL = 0, CONDITIONAL_WITH_DEFAULT = 1 };

    template <int BATCH, enum PlanePolicy PP, typename OpParamsType, typename DefaultType>
    struct BatchReadParams;

    template <int BATCH, typename OpParamsType, typename DefaultType>
    struct BatchReadParams<BATCH, CONDITIONAL_WITH_DEFAULT, OpParamsType, DefaultType> {
        OpParamsType op_params[BATCH];
        int usedPlanes;
        DefaultType default_value;
    };

    template <int BATCH, typename OpParamsType, typename DefaultType>
    struct BatchReadParams<BATCH, PROCESS_ALL, OpParamsType, DefaultType> {
        OpParamsType op_params[BATCH];
    };

    template <int BATCH, typename Operation>
    struct BatchReadCommon {
        protected:
        // BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION

        template <size_t Idx, typename Array>
        FK_HOST_FUSE auto get_element_at_index(const Array& array) -> decltype(array[Idx]) {
            return array[Idx];
        }

        template <size_t Idx, typename... Arrays>
        FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) {
            return Operation::build(get_element_at_index<Idx>(arrays)...);
        }

        template <size_t... Idx, typename... Arrays>
        FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, const Arrays&... arrays) {
            static_assert(sizeof...(Idx) == BATCH);
            using OutputArrayType = decltype(Operation::build(std::declval<typename Arrays::value_type>()...));
            return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... };
        }

        // END BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION
    };

    /*template <int BATCH>
    struct BatchReadCommon<BATCH, void> {};*/

    /// @brief struct BatchRead
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam Operation: the read Operation to perform on the data
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    template <int BATCH, typename Operation, enum PlanePolicy PP = PROCESS_ALL>
    struct BatchRead final : BatchReadCommon<BATCH, Operation> {
        using OutputType = typename Operation::OutputType;
        using ParamsType = BatchReadParams<BATCH, PP, typename Operation::ParamsType, OutputType>;
        using ReadDataType = typename Operation::ReadDataType;

        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ (PP == PROCESS_ALL) ? Operation::THREAD_FUSION : false };

    private:
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec_helper(const Point& thread, const typename Operation::ParamsType(&params)[BATCH]) {
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(thread, params[thread.z]);
            } else {
                return Operation::exec(thread, params[thread.z]);
            }
        }
    public:

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const ParamsType& params) {
            if constexpr (PP == CONDITIONAL_WITH_DEFAULT) {
                static_assert(ELEMS_PER_THREAD == 1, "ELEMS_PER_THREAD should be 1");
                if (params.usedPlanes <= thread.z) {
                    return params.default_value;
                } else {
                    return exec_helper<1>(thread, params.op_params);
                }
            } else {
                return exec_helper<ELEMS_PER_THREAD>(thread, params.op_params);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_x(thread, ptr.op_params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_y(thread, ptr.op_params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& ptr) {
            return Operation::num_elems_z(thread, ptr.op_params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.op_params[thread.z]);
        }
        using InstantiableType = Read<BatchRead<BATCH, Operation, PP>>;
        /// @brief build(): host function to create a Read instance of BatchRead(BATCH, Operation, PROCESS_ALL)
        /// @param params: is an array of parameters needed by each instance of the Operation
        /// @return: returns an instance of Read(BatchRead(BATCH, Operation, PROCESS_ALL))
        DEFAULT_READ_BUILD

        using InstantiableSourceType = SourceRead<BatchRead<BATCH, Operation, PP>>;
    private:
        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const int& usedPlanes, const OutputType& defaultValue, const std::integer_sequence<int, Idx...>&) {
            return { {{params[Idx]...}, usedPlanes, defaultValue} };
        }
        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{params[Idx]...}} };
        }
        // DEVICE FUNCTION BASED BUILDERS

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build_helper(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{instantiableOperations[Idx].params...}} };
        }

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build_helper(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const int& usedPlanes, const OutputType& defaultValue,
                     const std::integer_sequence<int, Idx...>&) {
            return { { {{ instantiableOperations[Idx].params... }, usedPlanes, defaultValue} } };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // STANDARD BUILDERS WITH PARAMS

        /// @brief build(): host function to create a Read instance of BatchRead with CONDITIONAL_WITH_DEFAULT
        /// @param params: is an array of parameters needed by each instance of the Operation
        /// @param usedPlanes: is the number of planes that need to execute the Operation::exec
        ///                    in order, from 0 to usedPlanes - 1
        /// @param defaultValue: is the value returned instead of Operation::exec for the planes
        ///                      whith thread.z index >= usedPlanes
        /// @return: returns an instance of Read(BatchRead(SIZE, Operation, CONDITIONAL_WITH_DEFAULT))
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build(const std::array<typename Operation::ParamsType, BATCH>& params,
              const int& usedPlanes, const OutputType& defaultValue) {
            return build_helper(params, output_size, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build(): host function to create a Read instance of BatchRead(BATCH, Operation, PROCESS_ALL)
        /// @param params: is an array of parameters needed by each instance of the Operation
        /// @return: returns an instance of Read(BatchRead(SIZE, Operation, CONDITIONAL_WITH_DEFAULT))
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build(const std::array<typename Operation::ParamsType, BATCH>& params) {
            return build_helper(params, output_size, std::make_integer_sequence<int, BATCH>{});
        }
        /// @brief build_source(): host function to create a ReadSource instance of
        /// BatchRead(BATCH, Operation, CONDITIONAL_WITH_DEFAULT)
        /// @param params: is an array of parameters needed by each instance of the Operation
        /// @param usedPlanes: is the number of planes that need to execute the Operation::exec
        ///                     in order, from 0 to usedPlanes - 1
        /// @param defaultValue: is the value returned instead of Operation::exec for the planes
        ///                      whith thread.z index >= usedPlanes
        /// @return: returns an instance of SourceRead(BatchRead(SIZE, Operation, CONDITIONAL_WITH_DEFAULT))
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableSourceType>
        build_source(const std::array<typename Operation::ParamsType, BATCH>& params, const int& usedPlanes,
                     const OutputType& defaultValue) {
            const InstantiableType instOperation = build(params, usedPlanes, defaultValue);
            // Assuming all the back functions have the same output size
            const Size output_size = Num_elems<InstantiableType>::size(Point(0, 0, 0), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width),
                                      static_cast<uint>(output_size.height),
                                      static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        /// @brief build_source(): host function to create a SourceRead instance of BatchRead(BATCH, Operation, PROCESS_ALL)
        /// @param params: is an array of parameters needed by each instance of the Operation
        /// @return: returns an instance of SourceRead(BatchRead(SIZE, Operation, PROCESS_ALL))
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
            std::enable_if_t<PP_ == PROCESS_ALL, InstantiableSourceType>
            build_source(const std::array<typename Operation::ParamsType, BATCH>& params) {
            const InstantiableType instOperation = build(params);
            // Assuming all the back functions have the same output size
            const Size outputSize = Num_elems<InstantiableType>::size(Point(0, 0, 0), instOperation);
            const dim3 activeThreads{ static_cast<uint>(outputSize.width), static_cast<uint>(outputSize.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        // END STANDARD BUILDERS WITH PARAMS

        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
            build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build(): CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
            build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                const int& usedPlanes, const OutputType& defaultValue) {
            return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build_source(): PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableSourceType>
            build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            const InstantiableType instOperation = build(instantiableOperations);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        /// @brief build_source(): CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableSourceType>
            build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                const int& usedPlanes, const OutputType& defaultValue) {
            const InstantiableType instOperation = build(instantiableOperations, usedPlanes, defaultValue);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        // END DEVICE FUNCTION BASED BUILDERS

        // BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION

        /// @brief 
        /// @tparam ...ArrayTypes 
        /// @param ...arrays 
        /// @return 
        template <typename... ArrayTypes, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
            build_batch(const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            const auto dfArray = build_helper_generic(std::make_index_sequence<BATCH>(), arrays...);
            return build(dfArray);
        }

        /// @brief 
        /// @tparam ...ArrayTypes 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @param ...arrays 
        /// @return 
        template <typename... ArrayTypes, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
            build_batch(const int& usedPlanes, const OutputType& defaultValue, const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            const auto dfArray = build_helper_generic(std::make_index_sequence<BATCH>(), arrays...);
            return build(dfArray, usedPlanes, defaultValue);
        }

        // END BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION
    };

    /// @brief struct BatchReadBack
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    /// @tparam Operation: the read Operation to perform on the data
    template <int BATCH, enum PlanePolicy PP = PROCESS_ALL, typename Operation = void>
    struct BatchReadBack final : BatchReadCommon<BATCH, Operation> {
        using InstanceType = ReadBackType;
        using OutputType = typename Operation::OutputType;
        using ParamsType = BatchReadParams<BATCH, PP, typename Operation::ParamsType, OutputType>;
        using BackFunction = typename Operation::BackFunction[BATCH];
        static constexpr bool THREAD_FUSION{ false };
        using ReadDataType = typename Operation::ReadDataType;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread,
                                      const ParamsType& params,
                                      const typename Operation::BackFunction(&back_function)[BATCH]) {
            if constexpr (PP == PROCESS_ALL) {
                return Operation::exec(thread, params.op_params[thread.z], back_function[thread.z]);
            } else {
                if (params.usedPlanes <= thread.z) {
                    return params.default_value;
                } else {
                    return Operation::exec(thread, params.op_params[thread.z], back_function[thread.z]);
                } 
            }
        }
        
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            return Operation::num_elems_x(thread, params.op_params[thread.z], back_function[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            return Operation::num_elems_y(thread, params.op_params[thread.z], back_function[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            return Operation::num_elems_z(thread, params.op_params[thread.z], back_function[thread.z]);
        }

        using InstantiableType = ReadBack<BatchReadBack<BATCH, PP, Operation>>;

        /// @brief 
        /// @param params 
        /// @param backFunction 
        /// @return 
        FK_HOST_FUSE 
        InstantiableType build(const ParamsType& params, const BackFunction& backFunction) {
            return InstantiableType{ { params, backFunction } };
        }

        using InstantiableSourceType = SourceReadBack<BatchReadBack<BATCH, PP, Operation>>;
    private:
        // STANDARD BUILDERS WITH PARAMS

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{{params[Idx]...}}, {back_functions[Idx]...}} };
        }

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build_helper(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                     const int& usedPlanes, const OutputType& defaultValue,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{{params[Idx]...}, usedPlanes, defaultValue}, {back_functions[Idx]...}} };
        }

        // END STANDARD BUILDERS WITH PARAMS

        // DEVICE FUNCTION BASED BUILDERS

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build_helper(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{instantiableOperations[Idx].params...}, {instantiableOperations[Idx].back_function...}} };
        }

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build_helper(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const int& usedPlanes, const OutputType& defaultValue,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{{ instantiableOperations[Idx].params... }, usedPlanes, defaultValue}, {instantiableOperations[Idx].back_function...}} };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // STANDARD BUILDERS WITH PARAMS

        /// @brief build():
        /// @param params
        /// @param back_functions
        /// @return
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build(const std::array<typename Operation::ParamsType, BATCH>& params,
              const std::array<typename Operation::BackFunction, BATCH>& back_functions) {
            return build_helper(params, back_functions, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief
        /// @param params
        /// @param back_functions
        /// @param usedPlanes
        /// @param defaultValue
        /// @return
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build(const std::array<typename Operation::ParamsType, BATCH>& params,
              const std::array<typename Operation::BackFunction, BATCH>& back_functions,
              const int& usedPlanes, const OutputType& defaultValue) {
            return build_helper(params, back_functions, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief 
        /// @param params 
        /// @param back_functions 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableSourceType>
        build_source(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const std::array<typename Operation::BackFunction, BATCH>& back_functions) {
            const InstantiableType instOperation = build(params, back_functions);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        /// @brief 
        /// @param params 
        /// @param back_functions 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableSourceType>
        build_source(const std::array<typename Operation::ParamsType, BATCH>& params,
                     const std::array<typename Operation::BackFunction, BATCH>& back_functions,
                     const int& usedPlanes, const OutputType& defaultValue) {
            const InstantiableType instOperation = build(params, back_functions, usedPlanes, defaultValue);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        // END STANDARD BUILDERS WITH PARAMS

        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build(): CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
              const int& usedPlanes, const OutputType& defaultValue) {
            return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build_source(): PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableSourceType>
        build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            const InstantiableType instOperation = build(instantiableOperations);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const dim3 activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH)};
            return make_source(instOperation, activeThreads);
        }

        /// @brief build_source(): CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableSourceType>
        build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const int& usedPlanes, const OutputType& defaultValue) {
            const InstantiableType instOperation = build(instantiableOperations, usedPlanes, defaultValue);
            const Size output_size = Num_elems<InstantiableType>::size(Point(), instOperation);
            const ActiveThreads activeThreads{ static_cast<uint>(output_size.width), static_cast<uint>(output_size.height), static_cast<uint>(BATCH) };
            return make_source(instOperation, activeThreads);
        }

        // END DEVICE FUNCTION BASED BUILDERS

        // BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION

        /// @brief 
        /// @tparam ...ArrayTypes 
        /// @param ...arrays 
        /// @return 
        template <typename... ArrayTypes, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
        build_batch(const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            const auto dfArray = build_helper_generic(std::make_index_sequence<BATCH>(), arrays...);
            return build(dfArray);
        }

        
        template <typename... ArrayTypes, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build_batch(const int& usedPlanes, const OutputType& defaultValue, const ArrayTypes&... arrays) {
            static_assert(allArraysSameSize_v<BATCH, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            const auto dfArray = build_helper_generic(std::make_index_sequence<BATCH>(), arrays...);
            return build(dfArray, usedPlanes, defaultValue);
        }

        // END BUILDERS THAT CAN USE ANY BUILDER FROM OPERATION
    };

    template <int BATCH, enum PlanePolicy PP>
    struct BatchReadBack<BATCH, PP, void> {
        // DEVICE FUNCTION BASED BUILDERS

        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == PROCESS_ALL, ReadBack<BatchReadBack<BATCH, PP_, Operation>>>
        build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return BatchReadBack<BATCH, PP, Operation>::build(instantiableOperations);
        }

        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, ReadBack<BatchReadBack<BATCH, PP_, Operation>>>
        build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
              const int& usedPlanes, const typename Operation::OutputType& defaultValue) {
            return BatchReadBack<BATCH, PP, Operation>::build(instantiableOperations, usedPlanes, defaultValue);
        }

        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE 
        std::enable_if_t<PP_ == PROCESS_ALL, SourceReadBack<BatchReadBack<BATCH, PP_, Operation>>>
        build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return BatchReadBack<BATCH, PP, Operation>::build_source(instantiableOperations);
        }

        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
        std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, SourceReadBack<BatchReadBack<BATCH, PP_, Operation>>>
        build_source(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                     const int& usedPlanes, const typename Operation::OutputType& defaultValue) {
            return BatchReadBack<BATCH, PP, Operation>::build_source(instantiableOperations, usedPlanes, defaultValue);
        }

        // END DEVICE FUNCTION BASED BUILDERS
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

} //namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
