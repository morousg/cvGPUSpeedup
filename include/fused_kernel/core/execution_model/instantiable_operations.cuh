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
#include <fused_kernel/core/execution_model/thread_fusion.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/array.cuh>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.cuh>
#include <fused_kernel/core/execution_model/vector_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/cast_base.cuh>

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

    enum PlanePolicy { PROCESS_ALL = 0, CONDITIONAL_WITH_DEFAULT = 1 };

    template <typename Operation_t>
    struct ReadInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)
    private:
        template <size_t BATCH, size_t... Idx, typename BackwardIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArrayBack(const std::index_sequence<Idx...>&,
                                          const std::array<BackwardIOp, BATCH>& bkArray,
                                          const std::array<ForwardIOp, BATCH>& fwdArray) {
            using ResultingType = decltype(ForwardIOp::Operation::build(std::declval<BackwardIOp>(), std::declval<ForwardIOp>()));
            return std::array<ResultingType, BATCH>{ForwardIOp::Operation::build(bkArray[Idx], fwdArray[Idx])...};
        }
        template <size_t BATCH, size_t... Idx, typename ThisIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArray(const std::index_sequence<Idx...>&,
                                          const std::array<ThisIOp, BATCH>& thisArray,
                                          const std::array<ForwardIOp, BATCH>& fwdArray) {
            using ResultingType = decltype(fuseDF(std::declval<ThisIOp>(), std::declval<ForwardIOp>()));
            return std::array<ResultingType, BATCH>{fuseDF(thisArray[Idx], fwdArray[Idx])...};
        }
        template <size_t BATCH, typename BackwardIOp, typename ForwardIOp, typename DefaultValueType>
        FK_HOST_FUSE auto then_helper(const std::array<BackwardIOp, BATCH>& backOpArray,
                                      const std::array<ForwardIOp, BATCH>& forwardOpArray,
                                      const int& usedPlanes,
                                      const DefaultValueType& defaultValue) {
            const auto fusedArray = make_fusedArrayBack<BATCH>(std::make_index_sequence<BATCH>{}, backOpArray, forwardOpArray);
            using ContinuationIOpNewType = typename decltype(fusedArray)::value_type;
            return ContinuationIOpNewType::Operation::build(usedPlanes, defaultValue, fusedArray);
        }
        template <size_t BATCH, typename BackwardIOp, typename ForwardIOp>
        FK_HOST_FUSE auto then_helper_2arr(const std::array<BackwardIOp, BATCH>& backOpArray,
                                           const std::array<ForwardIOp, BATCH>& forwardOpArray) {
            const auto fusedArray = make_fusedArrayBack<BATCH>(std::make_index_sequence<BATCH>{}, backOpArray, forwardOpArray);
            using ContinuationIOpNewType = typename decltype(fusedArray)::value_type;
            return ContinuationIOpNewType::Operation::build(fusedArray);
        }
        template <size_t BATCH, typename ThisIOp, typename ForwardIOp>
        FK_HOST_CNST auto then_helper(const std::array<ThisIOp, BATCH>& thisArray,
                                      const ForwardIOp& forwardIOp) const {
            const auto forwardOpArray = make_set_std_array<BATCH>(forwardIOp);
            const auto fusedArray = make_fusedArray<BATCH>(std::make_index_sequence<BATCH>{}, thisArray, forwardOpArray);
            using ContinuationIOpNewType = typename decltype(fusedArray)::value_type;
            if constexpr (Operation::PP == PROCESS_ALL) {
                return ContinuationIOpNewType::Operation::build(fusedArray);
            } else {
                using NewOutputType = typename ContinuationIOpNewType::Operation::OutputType;
                using OldOutputType = std::decay_t<decltype(this->params.default_value)>;
                const auto default_value = this->params.default_value;
                const auto val = UnaryV<CastBase<VBase<OldOutputType>, VBase<NewOutputType>>, OldOutputType, NewOutputType>::exec(default_value);
                return ContinuationIOpNewType::Operation::build(this->params.usedPlanes, val, fusedArray);
            }
        }
    public:
        template <typename ContinuationIOp>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            if constexpr (isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(Operation::BATCH == ContinuationIOp::Operation::BATCH,
                    "Fusing two batch operations of different BATCH size is not allowed.");
                static_assert(isReadBackType<typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                const auto backOpArray = Operation::toArray(*this);
                const auto forwardOpArray = ContinuationIOp::Operation::toArray(cIOp);
                if constexpr (Operation::PP == PROCESS_ALL && ContinuationIOp::Operation::PP == PROCESS_ALL) {
                    return then_helper_2arr<BATCH>(backOpArray, forwardOpArray);
                } else if constexpr (Operation::PP == PROCESS_ALL && ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                    return then_helper<BATCH>(backOpArray, forwardOpArray, cIOp.params.usedPlanes, cIOp.params.default_value);
                } else if constexpr (Operation::PP == CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == PROCESS_ALL) {
                    using BackType = std::decay_t<decltype(backOpArray)>;
                    using ForType = std::decay_t<decltype(forwardOpArray)>;
                    using FusedType = typename decltype(make_fusedArray<BATCH>(std::declval<std::make_index_sequence<BATCH>>(), std::declval<BackType>(), std::declval<ForType>()))::value_type;
                    using DefaultValueType = typename FusedType::Operation::OutputType;
                    if constexpr (std::is_same_v<typename Operation::OutputType, DefaultValueType>) {
                        return then_helper<BATCH>(backOpArray, forwardOpArray, this->params.usedPlanes, this->params.default_value);
                    } else {
                        using Original = typename BackType::value_type::Operation::OutputType;
                        const auto defaultValue = UnaryV<CastBase<VBase<Original>, VBase<DefaultValueType>>, Original, DefaultValueType>::exec(this->params.default_value);
                        return then_helper<BATCH>(backOpArray, forwardOpArray, this->params.usedPlanes, defaultValue);
                    }
                } else if constexpr (Operation::PP == CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                    if (this->params.usedPlanes != cIOp.params.usedPlanes) {
                        throw std::invalid_argument("Fusing two batch read operations with different number of used planes is not valid");
                    }
                    // We will take the default value of the continuation operation
                    return then_helper<BATCH>(backOpArray, forwardOpArray, cIOp.params.usedPlanes, cIOp.params.default_value);
                }
                
            } else if constexpr (!isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(isReadBackType<typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                const auto backOpArray = make_set_std_array<BATCH>(*this);
                const auto forwardOpArray = ContinuationIOp::Operation::toArray(cIOp);
                if constexpr (ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                    return then_helper<BATCH>(backOpArray, forwardOpArray, cIOp.params.usedPlanes, cIOp.params.default_value);
                } else {
                    return then_helper_2arr<BATCH>(backOpArray, forwardOpArray);
                }
            } else if constexpr (isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyReadType<ContinuationIOp> || isReadBackType<ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(Operation::BATCH);
                if constexpr (isReadBackType<ContinuationIOp>) {
                    const auto backOpArray = Operation::toArray(*this);
                    const auto forwardOpArray = make_set_std_array<BATCH>(cIOp);
                    if constexpr (Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                        return then_helper<BATCH>(backOpArray, forwardOpArray, this->params.usedPlanes, thsi->params.default_value);
                    } else {
                        return then_helper_2arr<BATCH>(backOpArray, forwardOpArray);
                    }
                } else {
                    const auto thisArray = Operation::toArray(*this);
                    return then_helper<BATCH>(thisArray, cIOp);
                }
            } else if constexpr (!isBatchOperation<Operation>&& !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyReadType<ContinuationIOp> || isReadBackType<ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                if constexpr (isReadBackType<ContinuationIOp>) {
                    return ContinuationIOp::Operation::build(*this, cIOp);
                } else {
                    return fuseDF(*this, cIOp);
                }
            }
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    template <typename Operation_t>
    struct ReadBackInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)
    private:
        template <size_t BATCH, size_t... Idx, typename BackwardIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArray(const std::index_sequence<Idx...>&,
                                          const std::array<BackwardIOp, BATCH>& bkArray,
                                          const std::array<ForwardIOp, BATCH>& fwdArray) {
            using ResultingType = decltype(ForwardIOp::Operation::build(std::declval<BackwardIOp>(), std::declval<ForwardIOp>()));
            return std::array<ResultingType, BATCH>{ForwardIOp::Operation::build(bkArray[Idx], fwdArray[Idx])...};
        }
        template <size_t BATCH, typename BackwardIOp, typename ForwardIOp, typename MainBatchIOp>
        FK_HOST_FUSE auto then_helper(const std::array<BackwardIOp, BATCH>& backOpArray,
                                      const std::array<ForwardIOp, BATCH>& forwardOpArray,
                                      const MainBatchIOp& mainIOp) {
            const auto fusedArray = make_fusedArray<BATCH>(std::make_index_sequence<BATCH>{}, backOpArray, forwardOpArray);
            using ContinuationIOpNewType = typename decltype(fusedArray)::value_type;
            if constexpr (MainBatchIOp::Operation::PP == PROCESS_ALL) {
                return ContinuationIOpNewType::Operation::build(fusedArray);
            } else {
                return ContinuationIOpNewType::Operation::build(mainIOp.params.usedPlanes, mainIOp.params.default_value, fusedArray);
            }
        }
    public:
        template <typename ContinuationIOp>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            if constexpr (isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(isReadBackType<typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                const auto backOpArray = make_set_std_array<BATCH>(*this);
                const auto forwardOpArray = ContinuationIOp::Operation::toArray(cIOp);
                return then_helper<BATCH>(backOpArray, forwardOpArray, cIOp);
            } else {
                static_assert(!isAnyReadType<ContinuationIOp> || isReadBackType<ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                if constexpr (isReadBackType<ContinuationIOp>) {
                    return ContinuationIOp::Operation::build(*this, cIOp);
                } else {
                    return fuseDF(*this, cIOp);
                }
            }
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    /**
    * @brief BinaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opDat)
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
    * @brief TernaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a IOp that will be used at some point in the implementation of the
    * Operation. It can be any kind of IOp.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opData)
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
    * @brief UnaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
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
    * @brief MidWriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another IOp can be executed after it, using the same data.
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
    * @brief WriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last IOp in a sequence of InstantiableOperations.
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

#include <fused_kernel/core/execution_model/default_builders_def.h>

    template <int BATCH, enum PlanePolicy PP, typename OpParamsType, typename DefaultType>
    struct BatchReadParams;

    template <int BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        int usedPlanes;
        DefaultType default_value;
    };

    template <int BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, PROCESS_ALL, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
    };

    template <int BATCH_, enum PlanePolicy PP__ = PROCESS_ALL, typename Operation_ = void, typename OutputType_ = NullType>
    struct BatchRead;

    /// @brief struct BatchRead
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam Operation: the read Operation to perform on the data
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    template <int BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, PROCESS_ALL, Operation_, OutputType_> {
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
        static constexpr PlanePolicy PP = PROCESS_ALL;
        using OutputType = typename Operation::OutputType;
        using ParamsType = BatchReadParams<BATCH, PROCESS_ALL, Operation, OutputType>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");
        using OperationDataType = OperationData<BatchRead<BATCH, PROCESS_ALL, Operation>>;
    private:
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec_helper(const Point& thread, const OperationData<Operation>(&opData)[BATCH]) {
            if constexpr (THREAD_FUSION) {
                return Operation::exec<ELEMS_PER_THREAD>(thread, opData[thread.z]);
            } else {
                return Operation::exec(thread, opData[thread.z]);
            }
        }
    public:

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const OperationData<BatchRead<BATCH, PROCESS_ALL, Operation>>& opData) {
            return exec_helper<ELEMS_PER_THREAD>(thread, opData.params.opData);
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
    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads_helper(const std::index_sequence<Idx...>&,
                                                                  const OperationDataType& opData) {
            return { cxp::max(num_elems_x(Point(0u, 0u, static_cast<uint>(Idx)), opData)...),
                     cxp::max(num_elems_y(Point(0u, 0u, static_cast<uint>(Idx)), opData)...),
                     static_cast<uint>(BATCH) };
        }
    public:
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return getActiveThreads_helper(std::make_index_sequence<BATCH>{}, opData);
        }
        using InstantiableType = Read<BatchRead<BATCH, PROCESS_ALL, Operation>>;
        DEFAULT_BUILD
    private:
        // DEVICE FUNCTION BASED BUILDERS

        template <int... Idx>
        FK_HOST_FUSE InstantiableType
        build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
                     const std::integer_sequence<int, Idx...>&) {
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...}}} };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): host function to create a Read instance PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <typename IOp>
        FK_HOST_FUSE InstantiableType build(const std::array<IOp, BATCH>& instantiableOperations) {
            static_assert(isAnyReadType<IOp>);
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }

        // END DEVICE FUNCTION BASED BUILDERS
        template <size_t... Idx>
        FK_HOST_FUSE std::array<Instantiable<Operation>, BATCH>
        toArray_helper(const std::index_sequence<Idx...>&, const Read<BatchRead<BATCH, PROCESS_ALL, Operation>>& mySelf) {
            return { Operation::build(mySelf.params.opData[Idx])... };
        }

        FK_HOST_FUSE std::array<Instantiable<Operation>, BATCH>
        toArray(const Read<BatchRead<BATCH, PROCESS_ALL, Operation>>& mySelf) {
            return toArray_helper(std::make_index_sequence<BATCH>{}, mySelf);
        }
    };

    template <int BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_> {
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
        static constexpr PlanePolicy PP = CONDITIONAL_WITH_DEFAULT;
        using OutputType = std::conditional_t<std::is_same_v<typename Operation::OutputType, NullType>, OutputType_, typename Operation::OutputType>;
        using ParamsType = BatchReadParams<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ false };
        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");
        using OperationDataType = OperationData<BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType_>>;
    public:

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const auto exec(const Point& thread, const OperationData<BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType_>>& opData) {
            if (opData.params.usedPlanes <= thread.z) {
                return opData.params.default_value;
            } else {
                return Operation::exec(thread, opData.params.opData[thread.z]);
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
    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads_helper(const std::index_sequence<Idx...>&,
            const OperationDataType& opData) {
            return { cxp::max(num_elems_x(Point(0u, 0u, static_cast<uint>(Idx)), opData)...),
                     cxp::max(num_elems_y(Point(0u, 0u, static_cast<uint>(Idx)), opData)...),
                     static_cast<uint>(BATCH) };
        }
    public:
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return getActiveThreads_helper(std::make_index_sequence<BATCH>{}, opData);
        }
        using InstantiableType = Read<BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType_>>;
        DEFAULT_BUILD
    private:
        // DEVICE FUNCTION BASED BUILDERS
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
                                                   const int& usedPlanes, const OutputType& defaultValue,
                                                   const std::integer_sequence<int, Idx...>&) {
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...}, usedPlanes, defaultValue}} };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // DEVICE FUNCTION BASED BUILDERS
        /// @brief build(): host function to create a Read instance CONDITIONAL_WITH_DEFAULT
        /// @param instantiableOperations 
        /// @param usedPlanes 
        /// @param defaultValue 
        /// @return 
        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations,
                                const int& usedPlanes, const DefaultValueType& defaultValue) {
            static_assert(isAnyReadType<IOp>);
            if constexpr (std::is_same_v<OutputType, NullType>) {
                return BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, typename IOp::Operation, DefaultValueType>::build_helper(instantiableOperations, usedPlanes, defaultValue,
                    std::make_integer_sequence<int, BATCH>{});
            } else {
                return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
            }
        }

        // END DEVICE FUNCTION BASED BUILDERS
        template <size_t... Idx>
        FK_HOST_FUSE std::array<Instantiable<Operation>, BATCH>
            toArray_helper(const std::index_sequence<Idx...>&, const Read<BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType_>>& mySelf) {
            return { Operation::build(mySelf.params.opData[Idx])... };
        }

        FK_HOST_FUSE std::array<Instantiable<Operation>, BATCH>
            toArray(const Read<BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, Operation, OutputType_>>& mySelf) {
            return toArray_helper(std::make_index_sequence<BATCH>{}, mySelf);
        }
    };

    template <int BATCH>
    struct BatchRead<BATCH, PROCESS_ALL, void> {
        template <typename IOp>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations) {
            return BatchRead<BATCH, PROCESS_ALL, typename IOp::Operation>::build(instantiableOperations);
        }
    };

    template <int BATCH>
    struct BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, void> {
        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations,
                                const int& usedPlanes, const DefaultValueType& defaultValue) {
            if constexpr (std::is_same_v<typename IOp::Operation::OutputType, NullType>) {
                return BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, typename IOp::Operation, DefaultValueType>::build(instantiableOperations, usedPlanes, defaultValue);
            } else {
                return BatchRead<BATCH, CONDITIONAL_WITH_DEFAULT, typename IOp::Operation>::build(instantiableOperations, usedPlanes, defaultValue);
            }
        }
    };

    template <int BATCH, typename Operation = void>
    struct BatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = typename Operation::ParamsType[BATCH];
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ Operation::THREAD_FUSION };
        using WriteDataType = typename Operation::WriteDataType;
        using OperationDataType = OperationData<BatchWrite<BATCH, Operation>>;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
            const ThreadFusionType<InputType, ELEMS_PER_THREAD>& input,
            const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                Operation::exec<ELEMS_PER_THREAD>(thread, input, opData.params[thread.z]);
            }
            else {
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
    private:
        // DEVICE FUNCTION BASED BUILDERS

        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& iOps,
                                                   const std::integer_sequence<int, Idx...>&) {
            return { {{(iOps[Idx].params)...}} };
        }

        // END DEVICE FUNCTION BASED BUILDERS
    public:
        // DEVICE FUNCTION BASED BUILDERS

        /// @brief build(): host function to create a Read instance PROCESS_ALL
        /// @param instantiableOperations 
        /// @return 
        template <typename IOp>
        FK_HOST_FUSE InstantiableType build(const std::array<IOp, BATCH>& iOps) {
            static_assert(isWriteType<IOp>);
            return build_helper(iOps, std::make_integer_sequence<int, BATCH>{});
        }
        // END DEVICE FUNCTION BASED BUILDERS
        DEFAULT_WRITE_BATCH_BUILD
    };

    template <int BATCH>
    struct BatchWrite<BATCH, void> {
        using InstaceType = WriteType;
        template <typename IOp>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& iOps) {
            return BatchWrite<BATCH, typename IOp::Operation>::build(iOps);
        }
    };

    namespace fused_operation_impl {
        // FusedOperation implementation struct
        template <typename Operation>
        FK_HOST_DEVICE_CNST typename Operation::OutputType
        exec_operate(const typename Operation::InputType& i_data) {
            return Operation::exec(i_data);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto
        exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            static_assert(isComputeType<Operation>, "The operation is WriteType and shouldn't be.");
            if constexpr (isUnaryType<Operation>) {
                return Operation::exec(i_data);
            } else {
                return Operation::exec(i_data, tuple.instance);
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread, const Tuple_& tuple) {
            return Tuple_::Operation::exec(thread, tuple.instance);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread,
                                              const typename Tuple_::Operation::InputType& i_data,
                                              const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (isComputeType<Operation> && !isUnaryType<Operation>) {
                return Operation::exec(i_data, tuple.instance);
            } else if constexpr (isUnaryType<Operation>) {
                return Operation::exec(i_data);
            } else if constexpr (isWriteType<Operation>) {
                // Assuming the behavior of a MidWriteType IOp
                Operation::exec(thread, i_data, tuple.instance);
                return i_data;
            } else if constexpr (isMidWriteType<Operation>) {
                // We are executing another FusedOperation that is MidWriteType
                return Operation::exec(thread, i_data, tuple.instance);
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

    template <typename Operation>
    struct FusedOperationOutputType<std::enable_if_t<isWriteType<Operation>>, Operation> {
        using type = typename Operation::InputType;
    };

    template <typename Operation>
    struct FusedOperationOutputType<std::enable_if_t<!isWriteType<Operation>>, Operation> {
        using type = typename Operation::OutputType;
    };

    template <typename... Operations>
    using FOOT = typename FusedOperationOutputType<void, Operations...>::type;

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
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input,
                                            const OperationDataType& opData) {
            return fused_operation_impl::tuple_operate(input, opData.params);
        }
        using InstantiableType = Binary<FusedOperation_<void, Operations...>>;
        DEFAULT_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        // In the future we can improve this by splitting the read op from the compute ops
        // in the TransformDPP
        static constexpr bool THREAD_FUSION{ sizeof...(Operations) > 1 ? false :
            FirstType_t<Operations...>::THREAD_FUSION };
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread,
                                            const OperationDataType& opData) {
            return fused_operation_impl::tuple_operate(thread, opData.params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_x(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_y(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_z(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        using InstantiableType = Read<FusedOperation_<void, Operations...>>;
        DEFAULT_BUILD
        DEFAULT_READ_BATCH_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isWriteType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const OperationDataType& opData) {
            return fused_operation_impl::tuple_operate(thread, input, opData.params);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        DEFAULT_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isMidWriteType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const OperationDataType& opData) {
            return fused_operation_impl::tuple_operate(thread, input, opData.params);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        DEFAULT_BUILD
    };

    template <typename... Operations>
    using FusedOperation = FusedOperation_<void, Operations...>;

#include <fused_kernel/core/execution_model/default_builders_undef.h>

    template <typename T>
    struct is_fused_operation : std::false_type {};

    template <typename... Operations>
    struct is_fused_operation<FusedOperation<Operations...>> : std::true_type {};

    template <typename IOp, typename Enabler = void>
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

    template <typename IOp>
    FK_HOST_DEVICE_CNST auto fusedOperationToOperationTuple(const IOp& df) {
        return InstantiableFusedOperationToOperationTuple<IOp>::value(df);
    }

    template <typename IOp>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const IOp& df) {
        using Op = typename IOp::Operation;
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

    template <typename IOp, typename... InstantiableOperations>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const IOp& df, const InstantiableOperations&... dfs) {
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

    /** @brief fuseDF: function that creates either a Read or a Binary IOp, composed of a
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
