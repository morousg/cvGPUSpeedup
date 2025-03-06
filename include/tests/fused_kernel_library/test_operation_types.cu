/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#ifdef WIN32
#include <intellisense/main.h>
#endif

#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cast.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>

// Operation types
// Read
using RPerThrFloat = fk::PerThreadRead<fk::_2D, float>;
// ReadBack
using RBResize = fk::ResizeRead<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, fk::Instantiable<RPerThrFloat>>;
// Unary
using UIntFloat = fk::Cast<int, float>;
using UFloatInt = fk::Cast<float, int>;
using Unaries = fk::TypeList<UIntFloat, UFloatInt>;
// Binary
using BAddInt = fk::Add<int>;
using BAddFloat = fk::Add<float>;
using Binaries = fk::TypeList<BAddInt, BAddFloat>;
// Ternary
using TInterpFloat = fk::Interpolate<fk::InterpolationType::INTER_LINEAR, fk::Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = fk::PerThreadWrite<fk::_2D, float>;
// MidWrite
using MWPerThrFloat = fk::FusedOperation<WPerThrFloat, BAddFloat>;

// Test combination type lists
template <typename... Types>
using TL = fk::TypeList<Types...>;

template <typename TL1, typename TL2>
using TLC = fk::TypeListCat_t<TL1, TL2>;

template <typename TL, typename T>
using ITB = fk::InsertTypeBack_t<TL, T>;

template <typename T, typename TL>
using ITF = fk::InsertTypeFront_t<T, TL>;

// No Read
using NoRead = ITB<ITB<ITB<TLC<TLC<TL<RBResize>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No ReadBack
using NoReadBack = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Unary
using NoUnary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Binary
using NoBinary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Ternary
using NoTernary = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, WPerThrFloat>, MWPerThrFloat>;
// No Write
using NoWrite = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, MWPerThrFloat>;
// No Midwrite
using NoMidWrite = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>;
// No AnyWrite
using NoAnyWrite = ITB<TLC<TLC<ITB<TL<RPerThrFloat>, RBResize>, Unaries>, Binaries>, TInterpFloat>;
// All Compute
using AllCompute = ITB<TLC<Unaries, Binaries>, TInterpFloat>;

template <typename TypeList>
struct IsReadType;
template <typename... Types>
struct IsReadType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::or_v<fk::isReadType<Types>...>;
};

template <typename TypeList>
struct IsReadBackType;
template <typename... Types>
struct IsReadBackType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::or_v<fk::isReadBackType<Types>...>;
};

template <typename TypeList>
struct NoneAnyWriteType;
template <typename... Types>
struct NoneAnyWriteType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::noneAnyWriteType<Types...>;
};

int launch() {
    // isReadType
    constexpr bool noneRead = !IsReadType<NoRead>::value;
    constexpr bool isRead = fk::isReadType<RPerThrFloat>;
    static_assert(noneRead && isRead, "Something wrong with isReadType");

    // isReadBackType
    constexpr bool noneReadBack = !IsReadBackType<NoReadBack>::value;
    constexpr bool isReadBack = fk::isReadBackType<RBResize>;
    static_assert(noneReadBack && isReadBack, "Something wrong with isReadType");

    // noneAnyWriteType
    constexpr bool noneAnyWriteType_v = NoneAnyWriteType<NoAnyWrite>::value;
    constexpr bool oneIsMidWrite = !NoneAnyWriteType<NoWrite>::value;
    constexpr bool oneIsWrite = !NoneAnyWriteType<NoMidWrite>::value;
    static_assert(fk::and_v<noneAnyWriteType_v, oneIsMidWrite, oneIsWrite>, "Something wrong with isReadType");

    return 0;
}
