/* Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_DEVIC_FUNCTIONS
#define FK_DEVIC_FUNCTIONS

#include <vector_types.h>
#include <fused_kernel/core/utils/operation_tuple.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/data/size.h>

namespace fk { // namespace FusedKernel

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

    template <typename Operation_t>
    struct ReadDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)
        static constexpr bool isSource{false};
        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief SourceReadDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * It uses the thread indexes, and an additional parameter which should contain the pointer or pointers from where to read.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params).
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct SourceReadDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)
        static constexpr bool isSource{ true };
        dim3 activeThreads;
        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    template <typename Operation_t>
    struct ReadBackDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)
        static constexpr bool isSource{ false };
        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief SourceReadBackDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * Additionally, it gets a DeviceFunction that it will use at some point of it's Operation implementation.
    * Usually, it will be another Read or ReadBack DeviceFunction.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function)
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct SourceReadBackDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)
        static constexpr bool isSource{ true };
        dim3 activeThreads;
        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief BinaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params)
    */
    template <typename Operation_t>
    struct BinaryDeviceFunction final : public OperationData<Operation_t> {
        using Operation = Operation_t;
        using InstanceType = BinaryType;
        IS_ASSERT(BinaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief TernaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a DeviceFunction that will be used at some point in the implementation of the
    * Operation. It can be any kind of DeviceFunction.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function)
    */
    template <typename Operation_t>
    struct TernaryDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(TernaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief UnaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename Operation_t>
    struct UnaryDeviceFunction {
        using Operation = Operation_t;
        using InstanceType = UnaryType;
        IS_ASSERT(UnaryType)

        template <typename... ContinuationsDF>
        FK_HOST_CNST auto then(const ContinuationsDF&... cDFs) const {
            return fuseDF(*this, cDFs...);
        }
    };

    /**
    * @brief MidWriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another DeviceFunction can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteDeviceFunction final : public OperationData<Operation_t> {
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
    * @brief WriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct WriteDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(WriteType)
    };

#undef DEVICE_FUNCTION_DETAILS_IS_ASSERT
#undef IS_ASSERT
#undef DEVICE_FUNCTION_DETAILS_IS
#undef DEVICE_FUNCTION_DETAILS
#undef ASSERT
#undef IS

    template <typename Operation>
    using SourceRead = SourceReadDeviceFunction<Operation>;
    template <typename Operation>
    using SourceReadBack = SourceReadBackDeviceFunction<Operation>;
    template <typename Operation>
    using Read = ReadDeviceFunction<Operation>;
    template <typename Operation>
    using ReadBack = ReadBackDeviceFunction<Operation>;
    template <typename Operation>
    using Unary = UnaryDeviceFunction<Operation>;
    template <typename Operation>
    using Binary = BinaryDeviceFunction<Operation>;
    template <typename Operation>
    using Ternary = TernaryDeviceFunction<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename Operation>
    using Write = WriteDeviceFunction<Operation>;

    template <typename DeviceFunction>
    FK_HOST_CNST auto make_source(const DeviceFunction& readDF, const dim3& activeThreads) {
        using Op = typename DeviceFunction::Operation;
        if constexpr (DeviceFunction::template is<ReadBackType>) {
            return SourceReadBack<Op>{{readDF.params, readDF.back_function}, activeThreads};
        } else {
            return SourceRead<Op>{{readDF.params}, activeThreads};
        }
    }

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
            // Assuming the behavior of a MidWriteType DeviceFunction
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
    using FOOT = FusedOperationOutputType<void, Operations...>::type;

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
            return tuple_operate<FirstOp, RemOps...>(input);
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
            return tuple_operate(input, tuple);
        }
        using InstantiableType = Binary<FusedOperation_<void, Operations...>>;
        DEFAULT_BINARY_BUILD
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<Operations...>;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        static constexpr bool THREAD_FUSION{ FirstType_t<Operations...>::THREAD_FUSION };

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return tuple_operate(thread, tuple);
        }
        using InstantiableType = Read<FusedOperation_<void, Operations...>>;
        DEFAULT_READ_BUILD
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
            return tuple_operate(thread, input, tuple);
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
            return tuple_operate(thread, input, tuple);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        DEFAULT_WRITE_BUILD
    };

    template <typename... Operations>
    using FusedOperation = FusedOperation_<void, Operations...>;

#include <fused_kernel/core/execution_model/default_builders_undef.h>

    template <typename T>
    struct is_fused_operation : std::false_type {};

    template <typename... Operations>
    struct is_fused_operation<FusedOperation<Operations...>> : std::true_type {};

    template <typename DeviceFunction, typename Enabler = void>
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

    template <typename DeviceFunction>
    FK_HOST_DEVICE_CNST auto fusedOperationToOperationTuple(const DeviceFunction& df) {
        return InstantiableFusedOperationToOperationTuple<DeviceFunction>::value(df);
    }

    template <typename DeviceFunction>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const DeviceFunction& df) {
        using Op = typename DeviceFunction::Operation;
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

    template <typename DeviceFunction, typename... DeviceFunctions>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const DeviceFunction& df, const DeviceFunctions&... dfs) {
        using Op = typename DeviceFunction::Operation;
        return cat(devicefunctions_to_operationtuple(df), devicefunctions_to_operationtuple(dfs...));
    }

    template <typename Operation, typename Enabler=void>
    struct DeviceFunctionType;

    // Single Operation cases
    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isReadType<Operation>>> {
        using type = Read<Operation>;
    };

    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isReadBackType<Operation>>> {
        using type = ReadBack<Operation>;
    };

    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isUnaryType<Operation>>> {
        using type = Unary<Operation>;
    };

    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isBinaryType<Operation>>> {
        using type = Binary<Operation>;
    };

    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isTernaryType<Operation>>> {
        using type = Ternary<Operation>;
    };

    template <typename Operation>
    struct DeviceFunctionType<Operation, std::enable_if_t<isWriteType<Operation>>> {
        using type = Write<Operation>;
    };

    template <typename Operation>
    using DF = typename DeviceFunctionType<Operation>::type;

    template <typename OperationTupleType, typename Enabler = void>
    struct OperationTupleToDeviceFunction;

    template <typename... Operations>
    struct OperationTupleToDeviceFunction<OperationTuple<Operations...>, std::enable_if_t<allUnaryTypes<Operations...>, void>> {
        static inline constexpr auto value(const OperationTuple<Operations...>& opTuple) {
            return DF<FusedOperation<Operations...>>{};
        }
    };

    template <typename... Operations>
    struct OperationTupleToDeviceFunction<OperationTuple<Operations...>, std::enable_if_t<!allUnaryTypes<Operations...>, void>> {
        static inline constexpr auto value(const OperationTuple<Operations...>& opTuple) {
            return DF<FusedOperation<Operations...>>{opTuple};
        }
    };

    template <typename OperationTuple>
    FK_HOST_CNST auto operationTuple_to_DeviceFunction(const OperationTuple& opTuple) {
        return OperationTupleToDeviceFunction<OperationTuple>::value(opTuple);
    }

    /** @brief fuseDF: function that creates either a Read or a Binary DeviceFunction, composed of an
    * OpertationTupleOperation (OTO), where the operations are the ones found in the DeviceFunctions in the
    * deviceFunctions parameter pack.
    * This is a convenience function to simplify the implementation of ReadBack and Ternary DeviceFunctions
    * and Operations.
    */
    template <typename... DeviceFunctions>
    FK_HOST_CNST auto fuseDF(const DeviceFunctions&... deviceFunctions) {
        using FirstDF = FirstType_t<DeviceFunctions...>;
        if constexpr (is_any_read_type<FirstDF>::value) {
            if constexpr (FirstDF::isSource) {
                return make_source(operationTuple_to_DeviceFunction(devicefunctions_to_operationtuple(deviceFunctions...)),
                    ppFirst(deviceFunctions...).activeThreads);
            } else {
                return operationTuple_to_DeviceFunction(devicefunctions_to_operationtuple(deviceFunctions...));
            }
        } else {
            return operationTuple_to_DeviceFunction(devicefunctions_to_operationtuple(deviceFunctions...));
        }
    }

    template <typename Operation, size_t NPtr, size_t... Index>
    constexpr inline std::array<DF<Operation>, NPtr> paramsArrayToDFArray_helper(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::index_sequence<Index...>&) 
    {
        return { (DF<Operation>{paramsArr[Index]})... };
    }

    template <typename Operation, size_t NPtr>
    constexpr inline std::array<DF<Operation>, NPtr> paramsArrayToDFArray(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr) 
    {
        return paramsArrayToDFArray_helper<Operation>(paramsArr, std::make_index_sequence<NPtr>{});
    }

    template <typename Operation, size_t NPtr, size_t... Index>
    constexpr inline std::array<DF<Operation>, NPtr> paramsArrayToDFArray_helper(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::array<typename Operation::BackFunction, NPtr>& backFunctions,
        const std::index_sequence<Index...>&) {
        return { (DF<Operation>{ OperationData<Operation>{paramsArr[Index], backFunctions[Index]} })... };
    }

    template <typename Operation, size_t NPtr>
    constexpr inline std::array<DF<Operation>, NPtr> paramsArrayToDFArray(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::array<typename Operation::BackFunction, NPtr>& backFunctions) {
        return paramsArrayToDFArray_helper<Operation>(paramsArr, backFunctions, std::make_index_sequence<NPtr>{});
    }

    template <typename T, size_t NPtr, size_t... Idx>
    constexpr inline std::array<Size, NPtr> ptrArrayToSizeArray_helper(std::index_sequence<Idx...>&, const std::array<RawPtr<_2D, T>, NPtr>& ptrs) {
        return { Size(static_cast<int>(ptrs[Idx].dims.width),
                      static_cast<int>(ptrs[Idx].dims.height))... };
    }

    template <typename T, size_t NPtr>
    constexpr inline std::array<Size, NPtr> ptrArrayToSizeArray(const std::array<RawPtr<_2D, T>, NPtr>& ptrs) {
        return ptrArrayToSizeArray_helper(std::make_index_sequence<NPtr>{}, ptrs);
    }

    template <typename, typename = std::void_t<>>
    struct has_operation : std::false_type {};
    
    template <typename T>
    struct has_operation<T, std::void_t<typename T::Operation>> : std::true_type {};
    
    template <typename, typename = std::void_t<>>
    struct has_instance_type : std::false_type {};
    
    template <typename T>
    struct has_instance_type<T, std::void_t<typename T::InstanceType>> : std::true_type {};

    template <typename T>
    struct is_device_function :
        std::conditional_t<has_operation<T>::value && has_instance_type<T>::value,
                           std::true_type, std::false_type> {};

    template <typename...>
    struct BackTypeHelper;

    // Base case: Only one Operation and ParamsArray
    template <typename Operation, typename ParamsArray>
    struct BackTypeHelper<TypeList<TypeList<Operation, ParamsArray>>> {
        using type = decltype(Operation::build(std::declval<typename ParamsArray::value_type>()));
    };

    // Recursive case: Multiple Operations and ParamsArrays
    template <typename Operation, typename ParamsArray, typename... Rest>
    struct BackTypeHelper<TypeList<TypeList<Operation, ParamsArray>, Rest...>> {
        using NextFinalType = typename BackTypeHelper<TypeList<Rest...>>::type;
        using type = decltype(Operation::build(
            std::declval<typename ParamsArray::value_type>(),
            std::declval<NextFinalType>()
        ));
    };

    template <typename OperationsAndArrays>
    using BackType = typename BackTypeHelper<OperationsAndArrays>::type;

} // namespace fk

#endif
