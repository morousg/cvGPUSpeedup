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

#ifndef FK_INSTANTIABLE_OPERATIONS
#define FK_INSTANTIABLE_OPERATIONS

#include <vector_types.h>
#include <fused_kernel/core/execution_model/operation_tuple.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/array.cuh>

namespace fk { // namespace FusedKernel

    struct ActiveThreads {
        uint x, y, z;
        FK_HOST_DEVICE_CNST ActiveThreads(const uint& vx = 1,
            const uint& vy = 1,
            const uint& vz = 1) : x(vx), y(vy), z(vz) {}
    };

#define IS \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

#define ASSERT(instance_type) \
    static_assert(std::is_same_v<typename Operation::InstanceType, instance_type>, "Operation is not " #instance_type );

#define DEVICE_FUNCTION_DETAILS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type;

#define DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    IS

#define IS_ASSERT(instance_type) \
    IS \
    ASSERT(instance_type)

#define DEVICE_FUNCTION_DETAILS_IS_ASSERT(instance_type) \
    DEVICE_FUNCTION_DETAILS(instance_type) \
    IS_ASSERT(instance_type)

    // Helper template to check for existence of static constexpr int BATCH 
    template <typename T, typename = void>
    struct has_batch : std::false_type {};
    template <typename T>
    struct has_batch<T, std::void_t<decltype(T::BATCH)>> : std::is_integral<decltype(T::BATCH)> {};
    // Helper template to check for existence of type alias Operation
    template <typename T, typename = void>
    struct has_operation : std::false_type {};
    template <typename T>
    struct has_operation<T, std::void_t<typename T::Operation>> : std::true_type {};
    // Combine checks into a single struct
    template <typename T> struct IsBatchOperation :
        std::integral_constant<bool, has_batch<T>::value && has_operation<T>::value> {};
    // Helper variable template
    template <typename T>
    constexpr bool isBatchOperation = IsBatchOperation<T>::value;

    template <typename Operation_t>
    struct ReadInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)

        template <typename ContinuationIOp>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            /*if constexpr (isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(Operation::BATCH == ContinuationIOp::Operation::BATCH,
                    "Fusing two batch operations of different BATCH size is not allowed.");

            } else if constexpr (!isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {

            } else if constexpr (isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {

            } else if constexpr (!isBatchOperation<Operation>&& !isBatchOperation<typename ContinuationIOp::Operation>) {*/
                if constexpr (isReadBackType<ContinuationIOp>) {
                    return ContinuationIOp::Operation::build(*this, cIOp);
                } else {
                    return fuseDF(*this, cIOp);
                }
            //}
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const ReadInstantiableOperation<Operation>& mySelf) {
            return { Operation::num_elems_x(Point(), mySelf.params),
                     Operation::num_elems_y(Point(), mySelf.params),
                     Operation::num_elems_z(Point(), mySelf.params)};
        }
    };

    template <typename Operation_t>
    struct ReadBackInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)

        template <typename ContinuationIOp>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            if constexpr (isReadBackType<ContinuationIOp>) {
                return ContinuationIOp::Operation::build(*this, cIOp);
            } else {
                return fuseDF(*this, cIOp);
            }
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const ReadBackInstantiableOperation<Operation>& mySelf) {
            return { Operation::num_elems_x(Point(), mySelf.params, mySelf.back_function),
                     Operation::num_elems_y(Point(), mySelf.params, mySelf.back_function),
                     Operation::num_elems_z(Point(), mySelf.params, mySelf.back_function)};
        }
    };

    /**
    * @brief BinaryInstantiableOperation: represents a InstantiableOperation that takes the result of the previous InstantiableOperation as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params)
    */
    template <typename Operation_t>
    struct BinaryInstantiableOperation final : public OperationData<Operation_t> {
        using Operation = Operation_t;
        using InstanceType = BinaryType;
        IS_ASSERT(BinaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief TernaryInstantiableOperation: represents a InstantiableOperation that takes the result of the previous InstantiableOperation as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a InstantiableOperation that will be used at some point in the implementation of the
    * Operation. It can be any kind of InstantiableOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function)
    */
    template <typename Operation_t>
    struct TernaryInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(TernaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief UnaryInstantiableOperation: represents a InstantiableOperation that takes the result of the previous InstantiableOperation as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename Operation_t>
    struct UnaryInstantiableOperation {
        using Operation = Operation_t;
        using InstanceType = UnaryType;
        IS_ASSERT(UnaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief MidWriteInstantiableOperation: represents a InstantiableOperation that takes the result of the previous InstantiableOperation as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another InstantiableOperation can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS(MidWriteType)
        static_assert(std::is_same_v<typename Operation::InstanceType, WriteType> ||
                      std::is_same_v<typename Operation::InstanceType, MidWriteType>,
                      "Operation is not WriteType or MidWriteType");

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief WriteInstantiableOperation: represents a InstantiableOperation that takes the result of the previous InstantiableOperation as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last InstantiableOperation in a sequence of InstantiableOperations.
    */
    template <typename Operation_t>
    struct WriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(WriteType)
    };

#undef DEVICE_FUNCTION_DETAILS_IS_ASSERT
#undef IS_ASSERT
#undef DEVICE_FUNCTION_DETAILS_IS
#undef DEVICE_FUNCTION_DETAILS
#undef ASSERT
#undef IS

    template <typename Operation>
    using Read = ReadInstantiableOperation<Operation>;
    template <typename Operation>
    using ReadBack = ReadBackInstantiableOperation<Operation>;
    template <typename Operation>
    using Unary = UnaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Binary = BinaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Ternary = TernaryInstantiableOperation<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteInstantiableOperation<Operation>;
    template <typename Operation>
    using Write = WriteInstantiableOperation<Operation>;

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

    /// @brief struct BatchRead
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam Operation: the read Operation to perform on the data
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    template <int BATCH_, enum PlanePolicy PP = PROCESS_ALL, typename Operation_ = void>
    struct BatchRead final : public BatchReadCommon<BATCH_, Operation_> {
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
        using OutputType = typename Operation::OutputType;
        using ParamsType = BatchReadParams<BATCH, PP, typename Operation::ParamsType, OutputType>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ (PP == PROCESS_ALL) ? Operation::THREAD_FUSION : false };
        static_assert(std::is_same_v<InstanceType, typename Operation::InstanceType>);

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
            return BATCH;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const ParamsType& ptr) {
            return Operation::pitch(thread, ptr.op_params[thread.z]);
        }
        static_assert(std::is_same_v<typename Operation::InstanceType, ReadType>, "Operation is not ReadType");
        using InstantiableType = Read<BatchRead<BATCH, PP, Operation>>;

    private:
        // DEVICE FUNCTION BASED BUILDERS

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
            build_helper(const std::array<Read<Operation>, BATCH>& instantiableOperations,
                const std::integer_sequence<int, Idx...>&) {
            return { {{instantiableOperations[Idx].params...}} };
        }

        template <int... Idx, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
            build_helper(const std::array<Read<Operation>, BATCH>& instantiableOperations,
                const int& usedPlanes, const OutputType& defaultValue,
                const std::integer_sequence<int, Idx...>&) {
            static_assert(std::is_same_v<typename InstantiableType::Operation::InstanceType, ReadType>);
            return { { {{ instantiableOperations[Idx].params... }, usedPlanes, defaultValue} } };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): host function to create a Read instance PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
            build(const std::array<Read<Operation>, BATCH>& instantiableOperations) {
            static_assert(std::is_same_v<InstanceType, typename Operation::InstanceType>);
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build(): host function to create a Read instance CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, InstantiableType>
        build(const std::array<Read<Operation>, BATCH>& instantiableOperations,
              const int& usedPlanes, const OutputType& defaultValue) {
            static_assert(std::is_same_v<InstanceType, typename Operation::InstanceType>);
            return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
        }

        // END DEVICE FUNCTION BASED BUILDERS
    };

    template <int BATCH, enum PlanePolicy PP>
    struct BatchRead<BATCH, PP, void> {
        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, Read<BatchRead<BATCH, PP_, Operation>>>
        build(const std::array<Read<Operation>, BATCH>& instantiableOperations) {
            return BatchRead<BATCH, PP, Operation>::build(instantiableOperations);
        }

        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, Read<BatchRead<BATCH, PP_, Operation>>>
        build(const std::array<Read<Operation>, BATCH>& instantiableOperations,
              const int& usedPlanes, const typename Operation::OutputType& defaultValue) {
            return BatchRead<BATCH, PP, Operation>::build(instantiableOperations, usedPlanes, defaultValue);
        }
    };

    /// @brief struct BatchReadBack
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    /// @tparam Operation: the read Operation to perform on the data
    template <int BATCH_, enum PlanePolicy PP = PROCESS_ALL, typename Operation_ = void>
    struct BatchReadBack final : public BatchReadCommon<BATCH_, Operation_> {
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
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
            return BATCH;
        }
        static_assert(std::is_same_v<typename Operation::InstanceType, ReadBackType>, "Operation is not ReadBackType");
        using InstantiableType = ReadBack<BatchReadBack<BATCH, PP, Operation>>;

    private:
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
        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): host function to create a Read instance PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE std::enable_if_t<PP_ == PROCESS_ALL, InstantiableType>
            build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }

        /// @brief build(): host function to create a Read instance CONDITIONAL_WITH_DEFAULT
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

        // END DEVICE FUNCTION BASED BUILDERS
    };

    template <int BATCH, enum PlanePolicy PP>
    struct BatchReadBack<BATCH, PP, void> {
        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): host function to create a Read instance
        /// @tparam Operation 
        /// @param instantiableOperations 
        /// @return 
        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
            std::enable_if_t<PP_ == PROCESS_ALL, ReadBack<BatchReadBack<BATCH, PP_, Operation>>>
            build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations) {
            return BatchReadBack<BATCH, PP, Operation>::build(instantiableOperations);
        }

        /// @brief build(): host function to create a Read instance
        /// @tparam Operation 
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <typename Operation, enum PlanePolicy PP_ = PP>
        FK_HOST_FUSE
            std::enable_if_t<PP_ == CONDITIONAL_WITH_DEFAULT, ReadBack<BatchReadBack<BATCH, PP_, Operation>>>
            build(const std::array<ReadBack<Operation>, BATCH>& instantiableOperations,
                const int& usedPlanes, const typename Operation::OutputType& defaultValue) {
            return BatchReadBack<BATCH, PP, Operation>::build(instantiableOperations, usedPlanes, defaultValue);
        }

        // END DEVICE FUNCTION BASED BUILDERS
    };

    namespace fused_operation_impl {
        // FusedOperation implementation struct
        template <typename Operation>
        FK_HOST_DEVICE_CNST typename Operation::OutputType exec_operate(const typename Operation::InputType& i_data) {
            return Operation::exec(i_data);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            static_assert(isComputeType<Operation>, "The operation is WriteType and shouldn't be.");
            if constexpr (std::is_same_v<typename Operation::InstanceType, TernaryType>) {
                return Operation::exec(i_data, tuple.instance.params, tuple.instance.back_function);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                return Operation::exec(i_data, tuple.instance.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                return Operation::exec(i_data);
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, ReadType>) {
                return Operation::exec(thread, tuple.instance.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, ReadBackType>) {
                return Operation::exec(thread, tuple.instance.params, tuple.instance.back_function);
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread,
            const typename Tuple_::Operation::InputType& i_data,
            const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, TernaryType>) {
                return Operation::exec(i_data, tuple.instance.params, tuple.instance.back_function);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                return Operation::exec(i_data, tuple.instance.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                return Operation::exec(i_data);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, WriteType>) {
                // Assuming the behavior of a MidWriteType InstantiableOperation
                Operation::exec(thread, i_data, tuple.instance.params);
                return i_data;
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, MidWriteType>) {
                // We are executing another FusedOperation that is MidWriteType
                return Operation::exec(thread, i_data, tuple.instance.params);
            }
        }

        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const typename FirstOp::InputType& i_data) {
            if constexpr (sizeof...(RemOps) == 0) {
                return FirstOp::exec(i_data);
            } else {
                return tuple_operate<RemOps...>(FirstOp::exec(i_data));
            }
        }

        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const typename FirstOp::InputType& i_data,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(i_data, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const Point& thread,
                                               const typename FirstOp::InputType& input,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, input, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(thread, result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const Point& thread,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(thread, result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
    } // namespace fused_operation_impl

    template <typename Enabler, typename... Operations>
    struct FusedOperationOutputType;

    template <typename... Operations>
    struct FusedOperationOutputType<std::enable_if_t<isWriteType<LastType_t<Operations...>>>, Operations...> {
        using type = typename LastType_t<Operations...>::InputType;
    };

    template <typename... Operations>
    struct FusedOperationOutputType<std::enable_if_t<!isWriteType<LastType_t<Operations...>>>, Operations...> {
        using type = typename LastType_t<Operations...>::OutputType;
    };

    template <typename... Operations>
    using FOOT = typename FusedOperationOutputType<void, Operations...>::type;

#include <fused_kernel/core/execution_model/default_builders_def.h>

    template <typename Enabler, typename... Operations>
    struct FusedOperation_ {};

    template <typename FirstOp, typename... RemOps>
    struct FusedOperation_<std::enable_if_t<allUnaryTypes<FirstOp, RemOps...> && (sizeof...(RemOps) + 1 > 1)>, FirstOp, RemOps...> {
        using InputType = typename FirstOp::InputType;
        using OutputType = typename LastType_t<RemOps...>::OutputType;
        using InstanceType = UnaryType;
        using Operations = TypeList<FirstOp, RemOps...>;

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return fused_operation_impl::tuple_operate<FirstOp, RemOps...>(input);
        }
        using InstantiableType = Unary<FusedOperation_<void, FirstOp, RemOps...>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename Operation>
    struct FusedOperation_<std::enable_if_t<isUnaryType<Operation>>, Operation> {
        using InputType = typename Operation::InputType;
        using OutputType = typename Operation::OutputType;
        using InstanceType = UnaryType;
        using Operations = TypeList<Operation>;

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return Operation::exec(input);
        }
        using InstantiableType = Unary<Operation>;
        DEFAULT_UNARY_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isComputeType<FirstType_t<Operations...>> &&
                           !allUnaryTypes<Operations...>>, Operations...> {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InstanceType = BinaryType;

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& tuple) {
            return fused_operation_impl::tuple_operate(input, tuple);
        }
        using InstantiableType = Binary<FusedOperation_<void, Operations...>>;
        DEFAULT_BINARY_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        static constexpr bool THREAD_FUSION{ sizeof...(Operations) > 1 ? false :
            FirstType_t<Operations...>::THREAD_FUSION };

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return fused_operation_impl::tuple_operate(thread, tuple);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_x(thread, params.instance.params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_y(thread, params.instance.params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_z(thread, params.instance.params);
        }

        using InstantiableType = Read<FusedOperation_<void, Operations...>>;
        DEFAULT_READ_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isReadBackType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        // In the future we can improve this by splitting the read op from the compute ops
        // in the TransformDPP
        static constexpr bool THREAD_FUSION{ sizeof...(Operations) > 1 ? false :
            FirstType_t<Operations...>::THREAD_FUSION };

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return fused_operation_impl::tuple_operate(thread, tuple);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_x(thread, params.instance.params, params.instance.back_function);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_y(thread, params.instance.params, params.instance.back_function);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const ParamsType& params) {
            return ParamsType::Operation::num_elems_z(thread, params.instance.params, params.instance.back_function);
        }

        using InstantiableType = Read<FusedOperation_<void, Operations...>>;
        DEFAULT_READ_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isWriteType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input, const ParamsType& tuple) {
            return fused_operation_impl::tuple_operate(thread, input, tuple);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isMidWriteType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input, const ParamsType& tuple) {
            return fused_operation_impl::tuple_operate(thread, input, tuple);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename... Operations>
    using FusedOperation = FusedOperation_<void, Operations...>;

#include <fused_kernel/core/execution_model/default_builders_undef.h>

    template <typename Operation, typename Enabler = void>
    struct InstantiableOperationType;

    // Single Operation cases
    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadType<Operation>>> {
        using type = Read<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadBackType<Operation>>> {
        using type = ReadBack<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isUnaryType<Operation>>> {
        using type = Unary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isBinaryType<Operation>>> {
        using type = Binary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isTernaryType<Operation>>> {
        using type = Ternary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isWriteType<Operation>>> {
        using type = Write<Operation>;
    };

    template <typename Operation>
    using Instantiable = typename InstantiableOperationType<Operation>::type;

    template <typename T>
    struct is_fused_operation : std::false_type {};

    template <typename... Operations>
    struct is_fused_operation<FusedOperation<Operations...>> : std::true_type {};

    template <typename InstantiableOperation, typename Enabler = void>
    struct InstantiableFusedOperationToOperationTuple;

    template <template <typename...> class SomeDF, typename... Operations>
    struct InstantiableFusedOperationToOperationTuple<SomeDF<FusedOperation<Operations...>>, std::enable_if_t<allUnaryTypes<Operations...>, void>> {
        FK_HOST_DEVICE_FUSE auto value(const SomeDF<FusedOperation<Operations...>>& df) {
            return OperationTuple<Operations...>{};
        }
    };
    template <template <typename...> class SomeDF, typename... Operations>
    struct InstantiableFusedOperationToOperationTuple<SomeDF<FusedOperation<Operations...>>, std::enable_if_t<!allUnaryTypes<Operations...>, void>> {
        FK_HOST_DEVICE_FUSE auto value(const SomeDF<FusedOperation<Operations...>>& df) {
            return df.params;
        }
    };

    template <typename InstantiableOperation>
    FK_HOST_DEVICE_CNST auto fusedOperationToOperationTuple(const InstantiableOperation& df) {
        return InstantiableFusedOperationToOperationTuple<InstantiableOperation>::value(df);
    }

    template <typename InstantiableOperation>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const InstantiableOperation& df) {
        using Op = typename InstantiableOperation::Operation;
        if constexpr (is_fused_operation<Op>::value) {
            return fusedOperationToOperationTuple(df);
        } else if constexpr (hasParamsAndBackFunction_v<Op>) {
            return OperationTuple<Op>{ {df.params, df.back_function} };
        } else if constexpr (hasParams_v<Op>) {
            return OperationTuple<Op>{ {df.params} };
        } else { // UnaryType case
            return OperationTuple<Op>{};
        }
    }

    template <typename InstantiableOperation, typename... InstantiableOperations>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const InstantiableOperation& df, const InstantiableOperations&... dfs) {
        using Op = typename InstantiableOperation::Operation;
        return cat(devicefunctions_to_operationtuple(df), devicefunctions_to_operationtuple(dfs...));
    }

    template <typename OperationTupleType, typename Enabler = void>
    struct OperationTupleToInstantiableOperation;

    template <typename... Operations>
    struct OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<allUnaryTypes<Operations...>, void>> {
        static inline constexpr auto value(const OperationTuple<Operations...>& opTuple) {
            return Instantiable<FusedOperation<Operations...>>{};
        }
    };

    template <typename... Operations>
    struct OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<!allUnaryTypes<Operations...>, void>> {
        static inline constexpr auto value(const OperationTuple<Operations...>& opTuple) {
            return Instantiable<FusedOperation<Operations...>>{opTuple};
        }
    };

    template <typename OperationTuple>
    FK_HOST_CNST auto operationTuple_to_InstantiableOperation(const OperationTuple& opTuple) {
        return OperationTupleToInstantiableOperation<OperationTuple>::value(opTuple);
    }

    /** @brief fuseDF: function that creates either a Read or a Binary InstantiableOperation, composed of a
    * FusedOperation, where the operations are the ones found in the InstantiableOperations in the
    * instantiableOperations parameter pack.
    * This is a convenience function to simplify the implementation of ReadBack and Ternary InstantiableOperations
    * and Operations.
    */
    template <typename... InstantiableOperations>
    FK_HOST_CNST auto fuseDF(const InstantiableOperations&... instantiableOperations) {
        return operationTuple_to_InstantiableOperation(devicefunctions_to_operationtuple(instantiableOperations...));
    }
} // namespace fk

#endif
