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
    enum ShiftDirection { Left, Right };

    template <typename T, ShiftDirection SD>
    struct Shift_ {
        BINARY_DECL_EXEC(T, T, uint) {
            static_assert(!validCUDAVec<T>, "Shift can't work with cuda vector types.");
            static_assert(std::is_unsigned_v<T>, "Shift only works with unsigned integers.");
            if constexpr (SD == Left) {
                return input << params;
            } else if constexpr (SD == Right) {
                return input >> params;
            }
        }
    };
    template <typename T, typename P, ShiftDirection SD>
    using Shift = BinaryV<Shift_<VBase<T>, SD>, T, P>;
    template <typename T, typename P = uint>
    using ShiftLeft = Shift<T, P, ShiftDirection::Left>;
    template <typename T, typename P = uint>
    using ShiftRight = Shift<T, P, ShiftDirection::Right>;

    template <typename I>
    struct IsEven {
        using InputType = I;
        using OutputType = bool;
        using InstanceType = UnaryType;
        using AcceptedTypes = TypeList<uchar, ushort, uint>;
        static constexpr __device__ __forceinline__ OutputType exec(const InputType& input) {
            static_assert(one_of_v<InputType, AcceptedTypes>, "Input type not valid for UnaryIsEven");
            return (input & 1u) == 0;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Max_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Max_ can't work with cuda vector types.");
            return input >= params ? input : params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Min_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Min_ can't work with cuda vector types.");
            return input <= params ? input : params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    using Max = BinaryV<Max_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
    template <typename I, typename P = I, typename O = I>
    using Min = BinaryV<Min_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
} //namespace fk
