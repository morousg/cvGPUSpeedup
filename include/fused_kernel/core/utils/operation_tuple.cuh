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

#include <fused_kernel/core/utils/tuple.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>

namespace fk {
    template <typename Operation>
    constexpr bool hasParams = one_of_v<typename Operation::InstanceType, TypeList<ReadType, BinaryType, WriteType, MidWriteType>>;

    template <typename Operation>
    constexpr bool hasParamsAndBackFunction = one_of_v<typename Operation::InstanceType, TypeList<ReadBackType, TernaryType>>;

    template <typename Enabler, typename... Operations>
    struct OperationTuple_;

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<hasParams<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<hasParamsAndBackFunction<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<hasParams<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        enum { size = 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<hasParamsAndBackFunction<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
        enum { size = 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        enum { size = 1 };
    };

    template <typename... Operations>
    using OperationTuple = OperationTuple_<void, Operations...>;

    template <int INDEX, typename... Instances>
    struct GetType<INDEX, OperationTuple_<void, Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto& get_params(OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get_params<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get_params<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.params;
            }
        } else {
            static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.params;
        }
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto get_params(const OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get_params<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get_params<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.params;
            }
        } else {
            static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.params;
        }
    }

    template <typename DeviceFunction>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const DeviceFunction& df) {
        using Op = typename DeviceFunction::Operation;
        if constexpr (hasParamsAndBackFunction<Op>) {
            return OperationTuple<Op>{ df.params, df.back_function };
        } else if constexpr (hasParams<Op>) {
            return OperationTuple<Op>{ df.params };
        } else { // UnaryType case
            return OperationTuple<Op>{};
        }
    }

    template <typename DeviceFunction, typename... DeviceFunctions>
    FK_HOST_DEVICE_CNST auto devicefunctions_to_operationtuple(const DeviceFunction& df, const DeviceFunctions&... dfs) {
        using Op = typename DeviceFunction::Operation;
        const OperationTuple<typename DeviceFunctions::Operation...> opTuple = devicefunctions_to_operationtuple(dfs...);
        if constexpr (hasParamsAndBackFunction<Op>) {
            return OperationTuple<Op, typename DeviceFunctions::Operation...>{ df.params, df.back_function, opTuple };
        } else if constexpr (hasParams<Op>) {
            return OperationTuple<Op, typename DeviceFunctions::Operation...>{ df.params, opTuple };
        } else { // UnaryType case
            return OperationTuple<Op, typename DeviceFunctions::Operation...>{ opTuple };
        }
    }

    template <typename Operation, typename Param>
    FK_HOST_DEVICE_CNST OperationTuple<Operation> make_operation_tuple(const Param& param) {
        static_assert(hasParams<Operation>, "Operation expected to have parameters");
        static_assert(std::is_same_v<typename Operation::ParamsType, Param>, "Operation ParamsType does not coincide with the parameter type, in make_operation_tuple");
        return { param };
    }

    template <typename FirstOperation, typename... Operations>
    FK_HOST_DEVICE_CNST OperationTuple<FirstOperation, Operations...> make_operation_tuple() {
        static_assert(!hasParams<FirstOperation>, "FirstOperation has parameters and it should not have.");
        if constexpr (sizeof...(Operations) > 0) {
            return { make_operation_tuple<Operations...>() };
        } else {
            return {};
        }
    }

    template <typename FirstOperation, typename... Operations, typename Param, typename... Params>
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParams<FirstOperation>, OperationTuple<FirstOperation, Operations...>>
        make_operation_tuple(const Param& param, const Params&... params) {
        using OpParamsType = typename FirstOperation::ParamsType;
        static_assert(std::is_same_v<OpParamsType, Param>, "FirstOperation ParamsType does not coincide with the first parameter type, in make_operation_tuple");
        return { param, make_operation_tuple<Operations...>(params...) };
    }

    template <typename FirstOperation, typename... Operations, typename Param, typename... Params>
    FK_HOST_DEVICE_CNST std::enable_if_t<!hasParams<FirstOperation>, OperationTuple<FirstOperation, Operations...>>
        make_operation_tuple(const Param& param, const Params&... params) {
        return { make_operation_tuple<Operations...>(param, params...) };
    }

    template <typename Type, typename = void>
    struct HasOperation : std::false_type {};

    template <typename Type>
    struct HasOperation<Type, std::void_t<typename Type::Operation>> : std::true_type {};

    struct IsDeviceFunction {
        template <typename Type>
        FK_HOST_DEVICE_FUSE bool complies() {
            return HasOperation<Type>::value;
        }
    };

    template <typename FirstOperation, typename... Operations, typename Param, typename BackFunction, typename... Params>
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParamsAndBackFunction<FirstOperation>,OperationTuple<FirstOperation, Operations...>> 
        make_operation_tuple(const Param& param, const BackFunction& back_function, const Params&... params) {
        static_assert(IsDeviceFunction::complies<BackFunction>(), "Expected a Device Funtion");
        return { param, back_function, make_operation_tuple<Operations...>(params...) };
    }

    template <typename Operation, typename Param, typename BackFunction>
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParamsAndBackFunction<Operation>, OperationTuple<Operation>>
        make_operation_tuple(const Param& param, const BackFunction& back_function) {
        static_assert(IsDeviceFunction::complies<BackFunction>(), "Expected a Device Funtion");
        return { param, back_function };
    }


} // namespace fk
