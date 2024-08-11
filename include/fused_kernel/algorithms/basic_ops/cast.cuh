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

#include <fused_kernel/core/execution_model/vector_operations.cuh>

namespace fk {
    template <typename I, typename O>
    struct CastBase {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            return static_cast<O>(input);
        }
    };

    template <typename I, typename O>
    struct Cast {
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            return UnaryV<I, O, CastBase<VBase<I>, VBase<O>>>::exec(input);
        }
    };
} // namespace fk
