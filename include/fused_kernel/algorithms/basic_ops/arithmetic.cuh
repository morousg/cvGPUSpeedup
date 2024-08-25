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

#include <fused_kernel/core/utils/tuple.cuh>
#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>

namespace fk {
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct Add {};

    template <typename I, typename P, typename O>
    struct Add<I, P, O, BinaryType> {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        static constexpr  __device__ __host__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct Add<I1, I2, O, UnaryType> {
        using InputType = Tuple<I1, I2>;
        using OutputType = O;
        using InstanceType = UnaryType;
        static constexpr  __device__ __host__ __forceinline__ OutputType exec(const InputType& input) {
            return get_v<0>(input) + get_v<1>(input);
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Sub {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        static constexpr  __device__ __host__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            return input - params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Mul {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        static constexpr  __device__ __host__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            return input * params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Div {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        static constexpr  __device__ __host__ __forceinline__ OutputType exec(const InputType& input, const ParamsType& params) {
            return input / params;
        }
    };
} // namespace fk

