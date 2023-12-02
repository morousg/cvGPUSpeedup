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
#include <fused_kernel/fusionable_operations/operations.cuh>

namespace fk { // namespace FusedKernel

#define DEVICE_FUNCTION_DETAILS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

    // generic operation structs
    template <typename Operation_t>
    struct ReadDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, ReadType>, "Operation is not Read.");
        DEVICE_FUNCTION_DETAILS(ReadType)
        typename Operation_t::ParamsType params;
        dim3 activeThreads;
    };

    template <typename Operation_t>
    struct BinaryDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, BinaryType>, "Operation is not Binary.");
        DEVICE_FUNCTION_DETAILS(BinaryType)
        typename Operation_t::ParamsType params;
    };

    template <typename... Operations>
    struct ComposedDeviceFunction {
        using Operation = ComposedOperationSequence<Operations...>;
        using InstanceType = BinaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };
        typename Operation::ParamsType params;
    };

    template <typename... Operations_t>
    struct UnaryDeviceFunction {
        using Operation = UnaryOperationSequence<Operations_t...>;
        using InstanceType = UnaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };
    };

    template <typename Operation_t>
    struct MidWriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        DEVICE_FUNCTION_DETAILS(MidWriteType)
        typename Operation_t::ParamsType params;
    };

    template <typename Operation_t>
    struct WriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        DEVICE_FUNCTION_DETAILS(WriteType)
        typename Operation_t::ParamsType params;
    };

#undef DEVICE_FUNCTION_DETAILS

    template <typename Operation>
    using Read = ReadDeviceFunction<Operation>;
    template <typename... Operations>
    using Unary = UnaryDeviceFunction<Operations...>;
    template <typename Operation>
    using Binary = BinaryDeviceFunction<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename... Operations>
    using Composed = ComposedDeviceFunction<Operations...>;
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
            template <typename DeviceFunction>
            FK_HOST_DEVICE_FUSE auto operate(const typename DeviceFunction::Operation::InputType& i_data,
                                             const DeviceFunction& deviceFunction) {
                if constexpr (DeviceFunction::template is<ReadType> || DeviceFunction::template is<BinaryType>) {
                    return DeviceFunction::Operation::exec(i_data, deviceFunction.params);
                } else if constexpr (DeviceFunction::template is<UnaryType>) {
                    return DeviceFunction::Operation::exec(i_data);
                } else if constexpr (DeviceFunction::template is<MidWriteType>) {
                    DeviceFunction::Operation::exec(i_data);
                    return i_data;
                }
            }

            template <typename I, typename Tuple>
            FK_HOST_DEVICE_FUSE OutputType apply_operate(const I& i_data,
                                                         const Tuple& deviceFunctionInstances) {
                if constexpr (thrust::tuple_size<Tuple>::value == 1) {
                    return operate(i_data, thrust::get<0>(deviceFunctionInstances));
                } else {
                    const auto [firstDF, restOfDF] = deviceFunctionInstances;
                    const auto result = operate(i_data, firstDF);
                    return apply_operate(result, restOfDF);
                }
            }

        public:
            FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input,
                                                const thrust::tuple<DeviceFunctionTypes...>& params) {
                return apply_operate(input, params);
            }
    };
} // namespace FusedKernel
