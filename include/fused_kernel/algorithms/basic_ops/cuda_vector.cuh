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

#include <fused_kernel/core/execution_model/operations.cuh>

namespace fk {
    template <typename I, typename O>
    struct Discard {
        using InputType = I; using OutputType = O; using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(cn<I> > cn<O>, "Output type should at least have one channel less");
            static_assert(std::is_same_v<typename VectorTraits<I>::base,
                typename VectorTraits<O>::base>,
                "Base types should be the same");
            if constexpr (cn<O> == 1) {
                if constexpr (std::is_aggregate_v<O>) {
                    return { input.x };
                } else {
                    return input.x;
                }
            } else if constexpr (cn<O> == 2) {
                return { input.x, input.y };
            } else if constexpr (cn<O> == 3) {
                return { input.x, input.y, input.z };
            }
        }
    };

    template <typename T, int... idxs>
    struct VectorReorder {
        using InputType = T;
        using OutputType = T;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(validCUDAVec<InputType>, "Non valid CUDA vetor type: UnaryVectorReorder");
            static_assert(cn<InputType> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
            return VReorder<idxs...>::exec(input);
        }
    };

    template <typename T, typename Operation>
    struct VectorReduce { 
        using InputType = T;
        using OutputType = VBase<T>;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            if constexpr (cn<T> == 2) {
                return Operation::exec(input.x, input.y);
            } else if constexpr (cn<T> == 3) {
                return Operation::exec(Operation::exec(input.x, input.y), input.z);
            } else if constexpr (cn<T> == 4) {
                return Operation::exec(Operation::exec(Operation::exec(input.x, input.y), input.z), input.w);
            }
        }
    };

    template <typename I, typename O>
    struct AddLast {
        using InputType = I;
        using OutputType = O;
        using ParamsType = typename VectorTraits<I>::base;
        using InstanceType = BinaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> == cn<OutputType> -1, "Output type should have one channel more");
            static_assert(std::is_same_v<typename VectorTraits<InputType>::base, typename VectorTraits<OutputType>::base>,
                "Base types should be the same");
            const ParamsType newElem = params;
            if constexpr (cn<InputType> == 1) {
                if constexpr (std::is_aggregate_v<InputType>) {
                    return { input.x, newElem };
                } else {
                    return { input, newElem };
                }
            } else if constexpr (cn<InputType> == 2) {
                return { input.x, input.y, newElem };
            } else if constexpr (cn<InputType> == 3) {
                return { input.x, input.y, input.z, newElem };
            }
        }
    };
}