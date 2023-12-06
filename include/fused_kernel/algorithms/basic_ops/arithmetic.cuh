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
    template <typename I, typename P = I, typename O = I>
    struct Sum_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sum_ can't work with cuda vector types.");
            return input + params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Sub_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Sub_ can't work with cuda vector types.");
            return input - params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Mul_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Mul_ can't work with cuda vector types.");
            return input * params;
        }
    };

    template <typename I, typename P = I, typename O = I>
    struct Div_ {
        BINARY_DECL_EXEC(O, I, P) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>, "Div_ can't work with cuda vector types.");
            return input / params;
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
    using Sum = BinaryV<Sum_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
    template <typename I, typename P = I, typename O = I>
    using Sub = BinaryV<Sub_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
    template <typename I, typename P = I, typename O = I>
    using Mul = BinaryV<Mul_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
    template <typename I, typename P = I, typename O = I>
    using Div = BinaryV<Div_<VBase<I>, VBase<P>, VBase<O>>, I, P, O>;
} // namespace fk

