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

#include <vector_types.h>

namespace fk { // namespace FusedKernel
    // generic operation structs
    template <typename Operation_t>
    struct ReadDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, ReadType>, "Operation is not Read.");
        typename Operation_t::ParamsType params;
        dim3 activeThreads;
        using Operation = Operation_t;
        using InstanceType = ReadType;
    };

    template <typename Operation_t>
    struct BinaryDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, BinaryType>, "Operation is not Binary.");
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
        using InstanceType = BinaryType;
    };

    template <typename Operation_t>
    struct UnaryDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, UnaryType>, "Operation is not Unary.");
        using Operation = Operation_t;
        using InstanceType = UnaryType;
    };

    template <typename Operation_t>
    struct MidWriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
        using InstanceType = MidWriteType;
    };

    template <typename Operation_t>
    struct WriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        typename Operation_t::ParamsType params;
        using Operation = Operation_t;
        using InstanceType = WriteType;
    };

    template <typename Operation>
    using Read = ReadDeviceFunction<Operation>;
    template <typename Operation>
    using Unary = UnaryDeviceFunction<Operation>;
    template <typename Operation>
    using Binary = BinaryDeviceFunction<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename Operation>
    using Write = WriteDeviceFunction<Operation>;

    // This is actually a Binary Operation, but it needs the DeviceFunctions definition to work
    template <typename... DeviceFunctionTypes>
    struct ComposedOperation {
        using InputType = FirstDeviceFunctionInputType_t<DeviceFunctionTypes...>;
        using ParamsType = thrust::tuple<DeviceFunctionTypes...>;
        using OutputType = LastDeviceFunctionOutputType_t<DeviceFunctionTypes...>;
        using InstanceType = BinaryType;
    private:
        template <typename Operation>
        FK_HOST_DEVICE_FUSE auto operate(const typename Operation::InputType& i_data,
                                         const Read<Operation>& df) {
            return Operation::exec(i_data, df.params);
        }

        template <typename Operation>
        FK_HOST_DEVICE_FUSE auto operate(const typename Operation::InputType& i_data,
                                               const Binary<Operation>& df) {
            return Operation::exec(i_data, df.params);
        }

        template <typename Operation>
        FK_HOST_DEVICE_FUSE auto operate(const typename Operation::InputType& i_data,
                                               const Unary<Operation>& df) {
            return Operation::exec(i_data);
        }

        template <typename I, typename Tuple>
        FK_HOST_DEVICE_FUSE OutputType apply_operate(const I& i_data,
                                                     const Tuple& deviceFunctionInstances) {
            if constexpr (thrust::tuple_size<Tuple>::value == 1) {
                return ComposedOperation<DeviceFunctionTypes...>::operate(i_data, thrust::get<0>(deviceFunctionInstances));
            } else {
                const auto [firstDF, restOfDF] = deviceFunctionInstances;
                const auto result = ComposedOperation<DeviceFunctionTypes...>::operate(i_data, firstDF);
                return ComposedOperation<DeviceFunctionTypes...>::apply_operate(result, restOfDF);
            }
        }

    public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input,
                                            const thrust::tuple<DeviceFunctionTypes...>& params) {
            return ComposedOperation<DeviceFunctionTypes...>::apply_operate(input, params);
        }
    };
} // namespace FusedKernel
