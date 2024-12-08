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

#pragma once

#include <fused_kernel/core/utils/operation_tuple.cuh>

namespace fk {
    // Primary template: assumes T does not have member 'next'
    template <typename, typename = std::void_t<>>
    struct has_next : std::false_type {};

    // Specialized template: this will be chosen if T has member 'next'
    template <typename T>
    struct has_next<T, std::void_t<decltype(std::declval<T>().next)>> : std::true_type {};

    // Helper variable template for easier usage
    template <typename T>
    constexpr bool has_next_v = has_next<T>::value;

    // FusedOperation implementation struct
    struct FusedOperationImpl {

    private:
        template <typename Operation>
        FK_HOST_DEVICE_FUSE typename Operation::OutputType exec_operate(const typename Operation::InputType& i_data) {
            return Operation::exec(i_data);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
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
        FK_HOST_DEVICE_FUSE auto exec_operate(const Point& thread, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, ReadType>) {
                return Operation::exec(thread, tuple.instance.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, ReadBackType>) {
                return Operation::exec(thread, tuple.instance.params, tuple.instance.back_function);
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto exec_operate(const Point& thread,
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

        template <typename FirstOp, typename... Operations>
        FK_HOST_DEVICE_FUSE
            auto tuple_operate_helper(const typename FirstOp::OutputType& result, const OperationTuple<FirstOp, Operations...>& tuple) {
            if constexpr (sizeof...(Operations) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, Operations...>>) {
                    return tuple_operate(result, tuple.next);
                } else {
                    return tuple_operate<Operations...>(result);
                }
            } else {
                return result;
            }
        }

        template <typename FirstOp, typename... Operations>
        FK_HOST_DEVICE_FUSE
            auto tuple_operate_helper(const Point& thread,
                                      const typename FirstOp::OutputType& result,
                                      const OperationTuple<FirstOp, Operations...>& tuple) {
            if constexpr (sizeof...(Operations) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, Operations...>>) {
                    return tuple_operate(thread, result, tuple.next);
                } else {
                    return tuple_operate<Operations...>(result);
                }
            } else {
                return result;
            }
        }

    public:
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const typename FirstOp::InputType& i_data) {
            const auto result = exec_operate<FirstOp>(i_data);
            if constexpr (sizeof...(RemOps) > 0) {
                static_assert(std::is_same_v<typename FirstOp::OutputType, typename FirstType_t<RemOps...>::InputType>,
                    "OutputType from current Operation does not match InputType from next Operation");
                return tuple_operate<RemOps...>(result);
            } else {
                return result;
            }
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const typename FirstOp::InputType& i_data,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(i_data, tuple);
            return tuple_operate_helper(result, tuple);
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const Point& thread,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, tuple);
            return tuple_operate_helper(thread, result, tuple);
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const Point& thread,
                                               const typename FirstOp::InputType& input,
                                               const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, input, tuple);
            return tuple_operate_helper(thread, result, tuple);
        }
    };

    
    
    template <typename Enabler, typename... Operations>
    struct FusedOperation_ {};

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<allUnaryTypes<Operations...>>, Operations...> {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = UnaryType;

    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return FusedOperationImpl::template tuple_operate<Operations...>(input);
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isComputeType<FirstType_t<Operations...>> &&
                                            !allUnaryTypes<Operations...>>, Operations...> {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = BinaryType;

    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& tuple) {
            return FusedOperationImpl::tuple_operate(input, tuple);
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        static constexpr bool THREAD_FUSION{ FirstType_t<Operations...>::THREAD_FUSION };
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return FusedOperationImpl::tuple_operate(thread, tuple);
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isWriteType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool THREAD_FUSION{ FirstType_t<Operations...>::THREAD_FUSION };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input, const ParamsType& tuple) {
            return FusedOperationImpl::tuple_operate(thread, input, tuple);
        }
    };

    template <typename... Operations>
    using FusedOperation = FusedOperation_<void, Operations...>;

    template <typename T>
    struct is_fused_operation : std::false_type {};

    template <typename... Operations>
    struct is_fused_operation<FusedOperation<Operations...>> : std::true_type {};

    template <template <typename...> class SomeDF, typename... Operations>
    FK_HOST_DEVICE_CNST auto fusedOperationToOperationTuple(const SomeDF<FusedOperation<Operations...>>& df) {
        if constexpr (hasParams_v<FusedOperation<Operations...>>) {
            return df.params;
        } else { // UnaryType case
            return OperationTuple<Operations...>{};
        }
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
} // namespace fk
