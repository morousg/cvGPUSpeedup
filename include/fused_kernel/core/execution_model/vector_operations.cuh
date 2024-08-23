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
#include <fused_kernel/core/utils/cuda_vector_utils.h>

namespace fk {
    template <typename I, typename O, typename Operation>
    struct UnaryV {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        static constexpr __device__ __host__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(cn<InputType> == cn<OutputType>, "Unary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA = (validCUDAVec<InputType> && validCUDAVec<OutputType>) ||
                !(validCUDAVec<InputType> || validCUDAVec<OutputType>);
            static_assert(allCUDAOrNotCUDA, "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<InputType> == 1) {
                if constexpr (validCUDAVec<InputType>) {
                    return { Operation::exec(input.x) };
                } else {
                    return Operation::exec(input);
                }
            } else if constexpr (cn<InputType> == 2) {
                return { Operation::exec(input.x),
                         Operation::exec(input.y) };

            } else if constexpr (cn<InputType> == 3) {
                return { Operation::exec(input.x),
                         Operation::exec(input.y),
                         Operation::exec(input.z) };
            } else {
                return { Operation::exec(input.x),
                         Operation::exec(input.y),
                         Operation::exec(input.z),
                         Operation::exec(input.w) };
            }
        }
    };

    template <typename Operation, typename I, typename P = I, typename O = I>
    struct BinaryV {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        static constexpr __device__ __host__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(cn<I> == cn<O>, "Binary struct requires same number of channels for input and output types.");
            constexpr bool allCUDAOrNotCUDA = (validCUDAVec<I> && validCUDAVec<O>) || !(validCUDAVec<I> || validCUDAVec<O>);
            static_assert(allCUDAOrNotCUDA, "Binary struct requires input and output types to be either both valild CUDA vectors or none.");

            if constexpr (cn<I> == 1) {
                if constexpr (validCUDAVec<I> && validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x) };
                } else if constexpr (validCUDAVec<I>) {
                    return { Operation::exec(input.x, params) };
                } else {
                    return Operation::exec(input, params);
                }
            } else if constexpr (cn<I> == 2) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params) };
                }

            } else if constexpr (cn<I> == 3) {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y),
                             Operation::exec(input.z, params.z) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params),
                             Operation::exec(input.z, params) };
                }

            } else {
                if constexpr (validCUDAVec<P>) {
                    return { Operation::exec(input.x, params.x),
                             Operation::exec(input.y, params.y),
                             Operation::exec(input.z, params.z),
                             Operation::exec(input.w, params.w) };
                } else {
                    return { Operation::exec(input.x, params),
                             Operation::exec(input.y, params),
                             Operation::exec(input.z, params),
                             Operation::exec(input.w, params) };
                }
            }
        }
    };
} // namespace fk
