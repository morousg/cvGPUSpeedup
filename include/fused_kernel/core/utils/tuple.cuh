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

#include <fused_kernel/core/utils/type_lists.cuh>

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
    struct Tuple {
        T instance;
        Tuple<Types...> next;
        enum { size = sizeof...(Types) + 1 };
    };

    struct TupleUtil {
    private:
        template <typename Tuple1, typename Tuple2, int... I1, int... I2>
        FK_HOST_DEVICE_FUSE auto cat_impl(const Tuple1&& t1, std::integer_sequence<int, I1...>,
                                          const Tuple2&& t2, std::integer_sequence<int, I2...>) {
            return make_tuple(Get<I1>::tuple_instance(t1)..., Get<I2>::tuple_instance(t2)...);
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

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(const Tuple1&& t1,
                                     const Tuple2&& t2) {
            return cat_impl(t1, std::make_integer_sequence<int, Tuple<Tuple1Types...>::size>(),
                            t2, std::make_integer_sequence<int, Tuple<Tuple2Types...>::size>());
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

    template <typename... Operations>
    struct OperationTuple {};

    template <typename Operation_t, typename... Operations>
    struct OperationTuple<Operation_t, Operations...> {
        using Enabler = std::enable_if_t<one_of_v<typename Operation_t::InstanceType, TypeList<ReadType, BinaryType, WriteType>>, int>;
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        OperationTuple<Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple<Operation_t> {
        using Enabler = std::enable_if_t<one_of_v<typename Operation_t::InstanceType, TypeList<ReadType, BinaryType, WriteType>>, int>;
        using Operation = Operation_t;
        typename Operation::ParamsType params;
        enum { size = 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple<Operation_t, Operations...> {
        using Enabler = std::enable_if_t<std::is_same_v<typename Operation_t::InstanceType, UnaryType>, int>;
        using Operation = Operation_t;
        OperationTuple<Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple<Operation_t> {
        using Enabler = std::enable_if_t<std::is_same_v<typename Operation_t::InstanceType, UnaryType>, int>;
        using Operation = Operation_t;
        enum { size = 1 };
    };


} // namespace fk

