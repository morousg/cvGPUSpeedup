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

#include <fused_kernel/core/utils/type_lists.cuh>
#include <fused_kernel/core/fusionable_operations/operations.cuh>

namespace fk {

    enum ColorConversionCodes {
        COLOR_BGR2BGRA = 0,
        COLOR_RGB2RGBA = COLOR_BGR2BGRA,
        COLOR_BGRA2BGR = 1,
        COLOR_RGBA2RGB = COLOR_BGRA2BGR,
        COLOR_BGR2RGBA = 2,
        COLOR_RGB2BGRA = COLOR_BGR2RGBA,
        COLOR_RGBA2BGR = 3,
        COLOR_BGRA2RGB = COLOR_RGBA2BGR,
        COLOR_BGR2RGB = 4,
        COLOR_RGB2BGR = COLOR_BGR2RGB,
        COLOR_BGRA2RGBA = 5,
        COLOR_RGBA2BGRA = COLOR_BGRA2RGBA,
        COLOR_BGR2GRAY = 6,
        COLOR_RGB2GRAY = 7,
        COLOR_BGRA2GRAY = 10,
        COLOR_RGBA2GRAY = 11
    };

    template <ColorConversionCodes value>
    using CCC_t = E_t<ColorConversionCodes, value>;

    using SupportedCCC = TypeList<CCC_t<COLOR_BGR2BGRA>,  CCC_t<COLOR_RGB2RGBA>,
                                  CCC_t<COLOR_BGRA2BGR>,  CCC_t<COLOR_RGBA2RGB>,
                                  CCC_t<COLOR_BGR2RGBA>,  CCC_t<COLOR_RGB2BGRA>,
                                  CCC_t<COLOR_BGRA2RGB>,  CCC_t<COLOR_RGBA2BGR>,
                                  CCC_t<COLOR_BGR2RGB>,   CCC_t<COLOR_RGB2BGR>,
                                  CCC_t<COLOR_BGRA2RGBA>, CCC_t<COLOR_RGBA2BGRA>,
                                  CCC_t<COLOR_RGB2GRAY>,  CCC_t<COLOR_RGBA2GRAY>,
                                  CCC_t<COLOR_BGR2GRAY>,  CCC_t<COLOR_BGRA2GRAY>>;

    template <ColorConversionCodes CODE>
    static constexpr bool isSuportedCCC = one_of_v<CCC_t<CODE>, SupportedCCC>;

    template <ColorConversionCodes CODE, typename I, typename O>
    struct ColorConversionType{
        static_assert(isSuportedCCC<CODE>, "Color conversion code not supported");
    };

    // Will work for COLOR_RGB2RGBA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2BGRA, I, O> {
        using type = Unary<AddAlpha<I>>;
    };

    // Will work for COLOR_RGBA2RGB too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2BGR, I, O> {
        using type = Unary<Discard<I, VectorType_t<VBase<I>, 3>>>;
    };

    // Will work for COLOR_RGB2BGRA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2RGBA, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>, AddAlpha<I>>;
    };

    // Will work for COLOR_RGBA2BGR too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2RGB, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>,
                           Discard<I, VectorType_t<VBase<I>, 3>>>;
    };

    // Will work for COLOR_RGB2BGR too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2RGB, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>>;
    };

    // Will work for COLOR_RGBA2BGRA too
    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2RGBA, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_RGB2GRAY, I, O> {
        using type = Unary<RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGR2GRAY, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0>, RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_RGBA2GRAY, I, O> {
        using type = Unary<RGB2Gray<I, O>>;
    };

    template <typename I, typename O>
    struct ColorConversionType<COLOR_BGRA2GRAY, I, O> {
        using type = Unary<VectorReorder<I, 2, 1, 0, 3>, RGB2Gray<I, O>>;
    };

    template <ColorConversionCodes code, typename I, typename O = I>
    using ColorConversion = typename ColorConversionType<code, I, O>::type;
}; // namespace fk (Fused Kernel)
