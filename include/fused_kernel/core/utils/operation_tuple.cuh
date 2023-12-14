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

#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/utils/tuple.cuh>

namespace fk {
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

    template <int INDEX>
    using OpTupUtils = OperationTupleUtils<INDEX>;
}
