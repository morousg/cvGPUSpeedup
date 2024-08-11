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
#include <fused_kernel/core/execution_model/operation_types.cuh>

namespace fk {
    template <typename... OperationTypes>
    struct UnaryOperationSequence {
        using InputType = typename FirstType_t<OperationTypes...>::InputType;
        using OutputType = typename LastType_t<OperationTypes...>::OutputType;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(std::is_same_v<typename FirstType_t<OperationTypes...>::InstanceType, UnaryType>);
            return UnaryOperationSequence<OperationTypes...>::next_exec<OperationTypes...>(input);
        }
    private:
        template <typename Operation>
        FK_HOST_DEVICE_FUSE typename Operation::OutputType next_exec(const typename Operation::InputType& input) {
            static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>);
            return Operation::exec(input);
        }
        template <typename Operation, typename... RemainingOperations>
        FK_HOST_DEVICE_FUSE typename LastType_t<RemainingOperations...>::OutputType next_exec(const typename Operation::InputType& input) {
            static_assert(std::is_same_v<typename Operation::InstanceType, UnaryType>);
            return UnaryOperationSequence<OperationTypes...>::next_exec<RemainingOperations...>(Operation::exec(input));
        }
    };
} //namespace fk
