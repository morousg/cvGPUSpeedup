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

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/execution_model/operation_types.cuh>

namespace fk {
    // Generic Tuple
    template <typename... Types>
    struct Tuple {};

    template <typename T>
    struct Tuple<T> {
        T instance;
        enum { size = 1 };
    };

    template <typename T, typename... Types>
    struct Tuple<T, Types...> {
        T instance;
        Tuple<Types...> next;
        enum { size = sizeof...(Types) + 1 };
    };

    struct TupleUtil {
    public:
        template <typename Tuple1, typename Tuple2, int... I1, int... I2>
        FK_HOST_DEVICE_FUSE auto cat_impl(const Tuple1& t1, std::integer_sequence<int, I1...>,
                                          const Tuple2& t2, std::integer_sequence<int, I2...>) {
            return make_tuple(get<I1>(t1)..., get<I2>(t2)...);
        }

    public:
        template <int INDEX, typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto& get(Tuple<InstanceTypes...>& instances) {
            constexpr int numberOfInstances = Tuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return get<INDEX - 1>(instances.next);
            } else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return get<numberOfInstances - 1>(instances.next);
                } else {
                    return instances.instance;
                }
            } else {
                return instances.instance;
            }
        }

        template <int INDEX, typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto get(const Tuple<InstanceTypes...>& instances) {
            constexpr int numberOfInstances = Tuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return get<INDEX - 1>(instances.next);
            } else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return get<numberOfInstances - 1>(instances.next);
                } else {
                    return instances.instance;
                }
            } else {
                return instances.instance;
            }
        }

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(Tuple1& t1, Tuple2& t2) {
            return cat_impl(t1, std::make_integer_sequence<int, Tuple1::size>(),
                            t2, std::make_integer_sequence<int, Tuple2::size>());
        }

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(const Tuple1& t1, const Tuple2& t2) {
            return cat_impl(t1, std::make_integer_sequence<int, Tuple1::size>(),
                            t2, std::make_integer_sequence<int, Tuple2::size>());
        }

        template <typename... Types>
        FK_HOST_DEVICE_FUSE auto make_tuple(const Types&... instances) {
            return Tuple<Types...>{instances...};
        }

        template <int INDEX, typename T, typename Tuple_>
        FK_HOST_DEVICE_FUSE auto tuple_insert(const T& instance, const Tuple_& tuple) {
            constexpr int numberOfInstances = Tuple_::size;
            static_assert(INDEX <= numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX == 0) {
                return TupleUtil::cat(make_tuple(instance), tuple);
            } else {
                if constexpr (Tuple_::size > 1) {
                    const auto [head, tail] = tuple;
                    return TupleUtil::cat(make_tuple(head), tuple_insert<INDEX - 1>(instance, tail));
                } else {
                    return TupleUtil::cat(tuple, make_tuple(instance));
                }
            }
        }
    };

    template <int INDEX, typename TupleLike>
    FK_HOST_DEVICE_CNST auto get_v(const TupleLike& tuple) {
        return fk::TupleUtil::get<INDEX>(tuple);
    }

    template <int INDEX, typename TupleLike>
    FK_HOST_DEVICE_CNST auto& get_v(TupleLike& tuple) {
        return fk::TupleUtil::get<INDEX>(tuple);
    }

    template <int INDEX, typename T, typename TupleLike>
    FK_HOST_DEVICE_CNST auto tuple_insert(const T& element, const TupleLike& tuple) {
        return fk::TupleUtil::tuple_insert<INDEX,T>(element, tuple);
    }

    template <typename Tuple1, typename Tuple2>
    FK_HOST_DEVICE_CNST auto cat(const Tuple1& t1, const Tuple2& t2) {
        return fk::TupleUtil::cat(t1, t2);
    }

    template <typename Operation>
    constexpr bool hasParams = one_of_v<typename Operation::InstanceType, TypeList<ReadType, BinaryType, WriteType, MidWriteType>>;

    template <typename Enabler, typename... Operations>
    struct OperationTuple_;

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<hasParams<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<hasParams<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        enum { size = 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<!hasParams<Operation_t>, void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<!hasParams<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        enum { size = 1 };
    };

    template <typename... Operations>
    using OperationTuple = OperationTuple_<void, Operations...>;

    template <int INDEX>
    struct OperationTupleUtils {
        template <typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto& get_params(OperationTuple<InstanceTypes...>& instances) {
            using Operation = typename OperationTuple<InstanceTypes...>::Operation;
            constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return OperationTupleUtils<INDEX - 1>::get_params(instances.next);
            }
            else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return OperationTupleUtils<numberOfInstances - 1>::get_params(instances.next);
                }
                else {
                    static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                    return instances.params;
                }
            }
            else {
                static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.params;
            }
        }

        template <typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto get_params(const OperationTuple<InstanceTypes...>& instances) {
            using Operation = typename OperationTuple<InstanceTypes...>::Operation;
            constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return OperationTupleUtils<INDEX - 1>::get_params(instances.next);
            } else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return OperationTupleUtils<numberOfInstances - 1>::get_params(instances.next);
                } else {
                    static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                    return instances.params;
                }
            } else {
                static_assert(hasParams<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.params;
            }
        }
    };

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto& get_params(OperationTuple<InstanceTypes...>& instances) {
        return OperationTupleUtils<INDEX>::get_params(instances);
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto get_params(const OperationTuple<InstanceTypes...>& instances) {
        return OperationTupleUtils<INDEX>::get_params(instances);
    }

    template <int INDEX>
    using OpTupUtils = OperationTupleUtils<INDEX>;

    template <int INDEX, typename TupleLike>
    struct tuple_element {};

    template <int INDEX, template <typename...> class TupleLike, typename... Instances>
    struct tuple_element<INDEX, TupleLike<Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <int INDEX, typename... Instances>
    struct tuple_element<INDEX, OperationTuple_<void, Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <int INDEX, typename TupleLike>
    using tuple_element_t = typename tuple_element<INDEX, TupleLike>::type;

} // namespace fk

