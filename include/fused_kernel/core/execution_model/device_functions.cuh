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
#include <utility>

namespace fk { // namespace FusedKernel

#define DEVICE_FUNCTION_DETAILS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

    /**
    * @brief ReadDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * It uses the thread indexes, and an additional parameter which should contain the pointer or pointers from where to read.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params).
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct ReadDeviceFunction {
        using Operation = Operation_t;
        static_assert(std::is_same_v<typename Operation::InstanceType, ReadType>, "Operation is not Read.");
        using InstanceType = ReadType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };

        typename Operation::ParamsType params;
        dim3 activeThreads;
    };

    /**
    * @brief ReadBackDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * Additionally, it gets a DeviceFunction that it will use at some point of it's Operation implementation.
    * Usually, it will be another Read or ReadBack DeviceFunction.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function)
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t, typename BackFunction>
    struct ReadBackDeviceFunction {
        using Operation = Operation_t;
        static_assert(std::is_same_v<typename Operation::InstanceType, ReadBackType>, "Operation is not ReadBack.");
        using InstanceType = ReadBackType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };

        typename Operation::ParamsType params;
        BackFunction back_function;
        dim3 activeThreads;
    };

    /**
    * @brief BinaryDeviceFunction_: implementation alias of BinaryDeviceFunction
    */
    template <typename Enabler, typename... Operations>
    struct BinaryDeviceFunction_ {};

    template <typename... Operations>
    struct BinaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) > 1)>, Operations...> {
        using Operation = OperationTupleOperation<Operations...>;
        using InstanceType = BinaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };

        constexpr BinaryDeviceFunction_() {}

        template <typename... ParamTypes>
        constexpr BinaryDeviceFunction_(const ParamTypes&... provided_params) : params(make_operation_tuple<Operations...>(provided_params...)) {}

        typename Operation::ParamsType params;
    };

    template <typename... Operations>
    struct BinaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) == 1)>, Operations...> {
        using Operation = FirstType_t<Operations...>;
        using InstanceType = BinaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };

        typename Operation::ParamsType params;
    };

    /**
    * @brief BinaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * OperationTupleOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params)
    */
    template <typename... Operations>
    using BinaryDeviceFunction = BinaryDeviceFunction_<void, Operations...>;

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
    template <typename Operation_t, typename BackFunction>
    struct TernaryDeviceFunction {
        using Operation = Operation_t;
        static_assert(std::is_same_v<typename Operation::InstanceType, TernaryType>, "Operation is not Ternary.");
        using InstanceType = TernaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };

        typename Operation::ParamsType params;
        BackFunction back_function;
    };

    /**
    * @brief UnaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename... Operations>
    struct UnaryDeviceFunction {
        using Operation = UnaryOperationSequence<Operations...>;
        static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>, "Operation is not Unary.");
        using InstanceType = UnaryType;
        template <typename IT>
        static constexpr bool is{ std::is_same_v<IT, InstanceType> };
    };

    /**
    * @brief MidWriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another DeviceFunction can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        DEVICE_FUNCTION_DETAILS(MidWriteType)

        typename Operation_t::ParamsType params;
    };

    /**
    * @brief WriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct WriteDeviceFunction {
        static_assert(std::is_same_v<typename Operation_t::InstanceType, WriteType>, "Operation is not Write.");
        DEVICE_FUNCTION_DETAILS(WriteType)

        typename Operation_t::ParamsType params;
    };

#undef DEVICE_FUNCTION_DETAILS

    template <typename Operation>
    using Read = ReadDeviceFunction<Operation>;
    template <typename ReadDFToReadBack, typename Operation_t>
    using ReadBack = ReadBackDeviceFunction<ReadDFToReadBack, Operation_t>;
    template <typename... Operations>
    using Unary = UnaryDeviceFunction<Operations...>;
    template <typename... Operations>
    using Binary = BinaryDeviceFunction<Operations...>;
    template <typename Operation, typename BackFunction>
    using Ternary = TernaryDeviceFunction<Operation, BackFunction>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename Operation>
    using Write = WriteDeviceFunction<Operation>;

    struct HasParams {
        template <typename Type>
        FK_HOST_DEVICE_FUSE bool complies() {
            return hasParams<Type>;
        }
    };

    template <size_t... Idx, typename... DeviceFunctions>
    FK_HOST_DEVICE_CNST auto fuseDF_helper(const std::index_sequence<Idx...>& idxSeq, const DeviceFunctions&... deviceFunctions) {
        if constexpr (FirstType_t<DeviceFunctions...>::template is<ReadType> ||
            FirstType_t<DeviceFunctions...>::template is<ReadBackType>) {
            const auto firstDF{ ppFirst(deviceFunctions...) };
            const dim3 activeThreads{ firstDF.activeThreads };
            return Read<OperationTupleOperation<typename DeviceFunctions::Operation...>> {
                make_operation_tuple<typename DeviceFunctions::Operation...>((ppGet<Idx>(deviceFunctions...).params)...),
                    activeThreads
            };
        } else {
            return Binary<OperationTupleOperation<typename DeviceFunctions::Operation...>> {
                make_operation_tuple<typename DeviceFunctions::Operation...>((ppGet<Idx>(deviceFunctions...).params)...)
            };
        }
    }

    template <typename... DeviceFunctions>
    FK_HOST_DEVICE_CNST auto fuseDF(const DeviceFunctions&... deviceFunctions) {
        using FilteredIdx = filtered_index_sequence_t<HasParams, TypeList<typename DeviceFunctions::Operation...>>;
        return fuseDF_helper(FilteredIdx{}, deviceFunctions...);
    }

} // namespace fk
