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

#include <fused_kernel/core/data/ptr_nd.h>

namespace fk {

    struct ReadType {};
    struct ReadBackType {};
    struct UnaryType {};
    struct BinaryType {};
    struct TernaryType {};
    struct MidWriteType {};
    struct WriteType {};

    template <typename OperationORDeviceFunction>
    constexpr bool isReadType = std::is_same_v<typename OperationORDeviceFunction::InstanceType, ReadType>;

    template <typename OperationORDeviceFunction>
    constexpr bool isReadBackType = std::is_same_v<typename OperationORDeviceFunction::InstanceType, ReadBackType>;

    using ReadTypeList = TypeList<ReadType, ReadBackType>;

    template <typename OperationORDeviceFunction>
    constexpr bool isAnyReadType = one_of_v<typename OperationORDeviceFunction::InstanceType, ReadTypeList>;

    template <typename OperationORDeviceFunction>
    constexpr bool isUnaryType = std::is_same_v<typename OperationORDeviceFunction::InstanceType, UnaryType>;

    template <typename OperationORDeviceFunction>
    constexpr bool isBinaryType = std::is_same_v<typename OperationORDeviceFunction::InstanceType, BinaryType>;

    template <typename OperationORDeviceFunction>
    constexpr bool isTernaryType = std::is_same_v<typename OperationORDeviceFunction::InstanceType, TernaryType>;

    using ComputeTypeList = TypeList<UnaryType, BinaryType, TernaryType>;

    template <typename OperationORDeviceFunction>
    constexpr bool isComputeType = one_of_v<typename OperationORDeviceFunction::InstanceType, ComputeTypeList>;

    template <typename DeviceFunction>
    using GetInputType_t = typename DeviceFunction::Operation::InputType;

    template <typename DeviceFunction>
    using GetOutputType_t = typename DeviceFunction::Operation::OutputType;

    template <typename DeviceFunction>
    FK_HOST_DEVICE_CNST GetOutputType_t<DeviceFunction> read(const Point& thread, const DeviceFunction& deviceFunction) {
        if constexpr (DeviceFunction::template is<ReadType>) {
            return DeviceFunction::Operation::exec(thread, deviceFunction.params);
        } else if constexpr (DeviceFunction::template is<ReadBackType>) {
            return DeviceFunction::Operation::exec(thread, deviceFunction.params, deviceFunction.back_function);
        }
    }

    template <typename DeviceFunction>
    FK_HOST_DEVICE_CNST GetOutputType_t<DeviceFunction> compute(const Point& thread,
        const GetInputType_t<DeviceFunction>& input,
        const DeviceFunction& deviceFunction) {
        static_assert(isComputeType<DeviceFunction>, "Function compute only works with DeviceFunction InstanceTypes of the group ComputeTypeList");
        if constexpr (isUnaryType<DeviceFunction>) {
            return DeviceFunction::Operation::exec(input);
        } else if constexpr (isBinaryType<DeviceFunction>) {
            return DeviceFunction::Operation::exec(input, deviceFunction.params);
        } else if constexpr (isTernaryType<DeviceFunction>) {
            return DeviceFunction::Operation::exec(input, deviceFunction.params, deviceFunction.back_function);
        }
    }

    template <typename... OperationsOrDeviceFunctions>
    constexpr bool allUnaryTypes = and_v<isUnaryType<OperationsOrDeviceFunctions>...>;

} // namespace fk
