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
#include <fused_kernel/core/execution_model/operations.cuh>

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

    template <typename... Operations>
    struct BinaryDeviceFunction {
        using Operation = OperationTupleOperation<Operations...>;
        using InstanceType = BinaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };
        typename Operation::ParamsType head;
    };

    template <typename... Operations>
    struct UnaryDeviceFunction {
        using Operation = UnaryOperationSequence<Operations...>;
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
    template <typename... Operations>
    using Binary = BinaryDeviceFunction<Operations...>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename Operation>
    using Write = WriteDeviceFunction<Operation>;
} // namespace FusedKernel
