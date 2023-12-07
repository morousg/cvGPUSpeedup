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

#include <fused_kernel/core/fusionable_operations/operations.cuh>

namespace fk {
    template <typename I, typename O>
    struct SaturateCast {
        UNARY_DECL_EXEC(I, O) {
            return saturate_cast<OutputType>(input);
        }
    };

    template <typename T>
    struct Saturate {
        using InputType = T;
        using OutputType = T;
        using ParamsType = VectorType_t<VBase<T>, 2>;
        using Base = typename VectorTraits<T>::base;
        using InstanceType = BinaryType;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& head) {
            static_assert(!validCUDAVec<T>, "Saturate only works with non cuda vector types");
            return Max<Base>::exec(head.params.x, Min<Base>::exec(input, head.params.y));
        }
    };

    template <typename T>
    struct SaturateFloat {
        using InputType = T;
        using OutputType = T;
        using Base = typename VectorTraits<T>::base;
        using InstanceType = UnaryType;

        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(std::is_same_v<Base, float>, "SaturateFloat only works with float types.");
            UnaryV<Saturate<float>, T>::exec(input, {0.f, 1.f});
        }
    };
}
