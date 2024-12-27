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

#ifndef FK_ARITHMETIC
#define FK_ARITHMETIC

#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct Add {};

    template <typename I, typename P, typename O>
    struct Add<I, P, O, BinaryType> {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
        using InstantiableType = Binary<Add<I, P, O, BinaryType>>;
        DEFAULT_BINARY_BUILD
    };

    template <typename I1, typename I2, typename O>
    struct Add<I1, I2, O, UnaryType> {
        using InputType = Tuple<I1, I2>;
        using OutputType = O;
        using InstanceType = UnaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) + get<1>(input);
        }
        using InstantiableType = Unary<Add<I1, I2, O, UnaryType>>;
        DEFAULT_UNARY_BUILD
    };

    template <typename I, typename P = I, typename O = I>
    struct Sub {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input - params;
        }
        using InstantiableType = Binary<Sub<I, P, O>>;
        DEFAULT_BINARY_BUILD
    };

    template <typename I, typename P = I, typename O = I>
    struct Mul {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input * params;
        }
        using InstantiableType = Binary<Mul<I, P, O>>;
        DEFAULT_BINARY_BUILD
    };

    template <typename I, typename P = I, typename O = I>
    struct Div {
        using OutputType = O;
        using InputType = I;
        using ParamsType = P;
        using InstanceType = BinaryType;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input / params;
        }
        using InstantiableType = Binary<Div<I, P, O>>;
        DEFAULT_BINARY_BUILD
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
