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
    struct OperationTupleOperationImpl {
    private:
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, TernaryType>) {
                return Operation::exec(i_data, tuple.params, tuple.back_function);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, BinaryType>) {
                return Operation::exec(i_data, tuple.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, UnaryType>) {
                return Operation::exec(i_data);
                // Assuming the behavior of a MidWriteType DeviceFunction
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, WriteType>) {
                Operation::exec(i_data, tuple.params);
                return i_data;
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto exec_operate(const Point& thread, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (std::is_same_v<typename Operation::InstanceType, ReadType>) {
                return Operation::exec(thread, tuple.params);
            } else if constexpr (std::is_same_v<typename Operation::InstanceType, ReadBackType>) {
                return Operation::exec(thread, tuple.params, tuple.back_function);
            }
        }
    public:
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const typename Tuple_::Operation::InputType& i_data,
            const Tuple_& tuple) {
            const auto result = exec_operate(i_data, tuple);
            if constexpr (Tuple_::size > 1) {
                return tuple_operate(result, tuple.next);
            } else {
                return result;
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_FUSE auto tuple_operate(const Point& thread, const Tuple_& tuple) {
            const auto result = exec_operate(thread, tuple);
            if constexpr (Tuple_::size > 1) {
                return tuple_operate(result, tuple.next);
            } else {
                return result;
            }
        }
    };
    using OTOImpl = OperationTupleOperationImpl;

    template <typename Enabler, typename... Operations>
    struct OperationTupleOperation_ {};

    template <typename... Operations>
    struct OperationTupleOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using InputType = typename FirstType_t<Operations...>::InputType;
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = BinaryType;

    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& tuple) {
            return OTOImpl::tuple_operate(input, tuple);
        }
    };

    template <typename... Operations>
    struct OperationTupleOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = typename LastType_t<Operations...>::OutputType;
        using InstanceType = ReadType;
        using ReadDataType = typename FirstType_t<Operations...>::ReadDataType;
        static constexpr bool THREAD_FUSION{ FirstType_t<Operations...>::THREAD_FUSION };
    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& tuple) {
            return OTOImpl::tuple_operate(thread, tuple);
        }
    };

    template <typename... Operations>
    using OperationTupleOperation = OperationTupleOperation_<void, Operations...>;
} // namespace fk
