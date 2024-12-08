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

#include <type_traits>

#include <fused_kernel/core/utils/tuple.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>

namespace fk {
    //Operation type traits
    // hasParams trait
    template <typename, typename = std::void_t<>>
    struct hasParams : std::false_type {};

    template <typename T>
    struct hasParams<T, std::void_t<typename T::ParamsType>> : std::true_type {};

    template <typename T>
    constexpr bool hasParams_v = hasParams<T>::value;

    using BFList = TypeList<ReadBackType, TernaryType>;
    template <typename OpOrDF>
    constexpr bool hasNoBackFunction_v = !one_of_v<typename OpOrDF::InstanceType, BFList>;

    // hasBackFunction trait
    template <typename, typename = std::void_t<>>
    struct hasBackFunction : std::false_type {};

    template <typename T>
    struct hasBackFunction<T, std::void_t<typename T::BackFunction>> : std::true_type {};

    template <typename T>
    constexpr bool hasBackFunction_v = hasBackFunction<T>::value;

    // hasParamsAndBackFunction trait
    template <typename, typename = std::void_t<>>
    struct hasParamsAndBackFunction : std::false_type {};

    template <typename T>
    struct hasParamsAndBackFunction<T, std::void_t<
        typename T::ParamsType,
        typename T::BackFunction>> : std::true_type {};

    template <typename T>
    constexpr bool hasParamsAndBackFunction_v = hasParamsAndBackFunction<T>::value;

    // OperationData implementation selectors
    template <typename Operation>
    constexpr bool hasParamsNoArray =
        hasParams_v<Operation> && !std::is_array_v<typename Operation::ParamsType>;
    template <typename Operation>
    constexpr bool hasParamsArray =
        hasParams_v<Operation> && std::is_array_v<typename Operation::ParamsType>;

    // OperationData implementations
    template <typename Operation, typename Enabler=void> struct OperationData;

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsNoArray<Operation> && hasNoBackFunction_v<Operation>, void>> {
        constexpr OperationData() {};
        constexpr OperationData(const typename Operation::ParamsType& params_) : params(params_) {}
        typename Operation::ParamsType params;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsArray<Operation> && hasNoBackFunction_v<Operation>, void>> {
        OperationData() {};
        OperationData(const typename Operation::ParamsType& params_) {
            std::copy(std::begin(params_), std::end(params_), std::begin(params));
        }
        OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
            }
            return *this;
        }
        typename Operation::ParamsType params;
    };

    template <typename Operation> 
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackFunction_v<Operation> &&
        !std::is_array_v<typename Operation::ParamsType> &&
        !std::is_array_v<typename Operation::BackFunction>, void>> {
        constexpr OperationData() {};
        constexpr OperationData(const typename Operation::ParamsType& params_,
                                const typename Operation::BackFunction& back_function_) :
            params(params_), back_function(back_function_) {}
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackFunction_v<Operation> &&
        (std::is_array_v<typename Operation::ParamsType> ||
            std::is_array_v<typename Operation::BackFunction>), void>> {
        OperationData() {};
        OperationData(const typename Operation::ParamsType& params_,
            const typename Operation::BackFunction& back_function_) {
            if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                std::copy(std::begin(params_), std::end(params_), std::begin(params));
            } else {
                params = params_;
            }
            if constexpr (std::is_array_v<typename Operation::BackFunction>) {
                std::copy(std::begin(back_function_), std::end(back_function_), std::begin(back_function));
            } else {
                back_function = back_function_;
            }
        }
        OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                    std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
                } else {
                    params = other.params;
                }
                if constexpr (std::is_array_v<typename Operation::BackFunction>) {
                    std::copy(std::begin(other.back_function), std::end(other.back_function), std::begin(back_function));
                } else {
                    back_function = other.back_function;
                }
            }
            return *this;
        }
        typename Operation::ParamsType params;
        typename Operation::BackFunction back_function;
    };

    template <typename Enabler, typename... Operations>
    struct OperationTuple_;

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<!isUnaryType<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        OperationData<Operation> instance;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t> && !allUnaryTypes<Operations...>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<allUnaryTypes<Operation_t, Operations...>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<!isUnaryType<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        OperationData<Operation> instance;
        enum { size = 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        enum { size = 1 };
    };

    template <typename... Operations>
    using OperationTuple = OperationTuple_<void, Operations...>;

    template <typename... Operations>
    constexpr bool allOpTupleUnary_f(const OperationTuple<Operations...>& opTup) {
        return allUnaryTypes<Operations...>;
    }

    template <typename OpTuple>
    constexpr bool allOpTupleUnary = allOpTupleUnary_f(OpTuple{});

    template <int INDEX, typename... Instances>
    struct GetType<INDEX, OperationTuple_<void, Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <typename... Operations, typename... OperationDatas>
    FK_HOST_DEVICE_CNST OperationTuple<Operations...> make_operation_tuple_(const OperationDatas&... instances) {
        return OperationTuple<Operations...>{instances...};
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto& get(OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.instance;
            }
        } else {
            static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.instance;
        }
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto get(const OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.instance;
            }
        } else {
            static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.instance;
        }
    }

    struct NotUnaryRestriction {
        template <typename Type>
        static constexpr __host__ __device__ bool complies() {
            using NotUnary =
                TypeList<ReadType, ReadBackType, BinaryType, TernaryType, MidWriteType, WriteType>;
            if constexpr (one_of_v<Type, NotUnary>) {
                return true;
            } else {
                return false;
            }
        }
    };

    template <typename... Operations1, typename... Operations2, int... I1, int... I2>
    FK_HOST_DEVICE_CNST auto cat_impl(const OperationTuple<Operations1...>& t1, const std::integer_sequence<int, I1...>& is1,
                                      const OperationTuple<Operations2...>& t2, const std::integer_sequence<int, I2...>& is2) {
        return make_operation_tuple_<Operations1..., Operations2...>(get<I1>(t1)..., get<I2>(t2)...);
    }

    template <typename... Operations1, typename... Operations2>
    FK_HOST_DEVICE_CNST auto cat(OperationTuple<Operations1...>& t1, OperationTuple<Operations2...>& t2) {
        return cat_impl(t1, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations1::InstanceType...>>{},
                        t2, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations2::InstanceType...>>{});
    }

    template <typename... Operations1, typename... Operations2>
    FK_HOST_DEVICE_CNST auto cat(const OperationTuple<Operations1...>& t1, const OperationTuple<Operations2...>& t2) {
        return cat_impl(t1, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations1::InstanceType...>>{},
                        t2, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations2::InstanceType...>>{});
    }

    template <typename Operation, typename Param>
    FK_HOST_DEVICE_CNST OperationTuple<Operation> make_operation_tuple(const Param& param) {
        static_assert(hasParams_v<Operation>, "Operation expected to have parameters");
        static_assert(std::is_same_v<typename Operation::ParamsType, Param>, "Operation ParamsType does not coincide with the parameter type, in make_operation_tuple");
        return { {param} };
    }

    template <typename FirstOperation, typename... Operations>
    FK_HOST_DEVICE_CNST OperationTuple<FirstOperation, Operations...> make_operation_tuple() {
        static_assert(!hasParams_v<FirstOperation>, "FirstOperation has parameters and it should not have.");
        if constexpr (sizeof...(Operations) > 0) {
            return { make_operation_tuple<Operations...>() };
        } else {
            return {};
        }
    }

    template <typename FirstOperation, typename... Operations, typename Param, typename... Params>
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParams_v<FirstOperation>, OperationTuple<FirstOperation, Operations...>>
        make_operation_tuple(const Param& param, const Params&... params) {
        using OpParamsType = typename FirstOperation::ParamsType;
        static_assert(std::is_same_v<OpParamsType, Param>, "FirstOperation ParamsType does not coincide with the first parameter type, in make_operation_tuple");
        return { param, make_operation_tuple<Operations...>(params...) };
    }

    template <typename FirstOperation, typename... Operations, typename Param, typename... Params>
    FK_HOST_DEVICE_CNST std::enable_if_t<!hasParams_v<FirstOperation>, OperationTuple<FirstOperation, Operations...>>
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
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParamsAndBackFunction_v<FirstOperation>,OperationTuple<FirstOperation, Operations...>> 
        make_operation_tuple(const Param& param, const BackFunction& back_function, const Params&... params) {
        static_assert(IsDeviceFunction::complies<BackFunction>(), "Expected a Device Funtion");
        return { {param, back_function}, make_operation_tuple<Operations...>(params...) };
    }

    template <typename Operation, typename Param, typename BackFunction>
    FK_HOST_DEVICE_CNST std::enable_if_t<hasParamsAndBackFunction_v<Operation>, OperationTuple<Operation>>
        make_operation_tuple(const Param& param, const BackFunction& back_function) {
        static_assert(IsDeviceFunction::complies<BackFunction>(), "Expected a Device Funtion");
        return { {param, back_function} };
    }


} // namespace fk
