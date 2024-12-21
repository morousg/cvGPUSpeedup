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

#include "tests/main.h"

#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cast.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/utils/type_lists.h>

// Operation types
// Read
using RPerThrFloat = fk::PerThreadRead<fk::_2D, float>;
// ReadBack
using RBResize = fk::ResizeRead<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, fk::Instantiable<RPerThrFloat>>;
// Unary
using UIntFloat = fk::Cast<int, float>;
using UFloatInt = fk::Cast<float, int>;
// Binary
using BAddInt = fk::Add<int>;
using BAddFloat = fk::Add<float>;
// Ternary
using TInterpFloat = fk::Interpolate<fk::InterpolationType::INTER_LINEAR, fk::Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = fk::PerThreadWrite<fk::_2D, float>;
// MidWrite
using MWPerThrFloat = fk::FusedOperation<WPerThrFloat, BAddFloat>;

int launch() {
    constexpr fk::Instantiable<RPerThrFloat> func1{};
    constexpr auto func2 =
        func1.then(fk::Instantiable<UFloatInt>{}).
        then(fk::Instantiable<BAddInt>{4}).
        then(fk::Instantiable<UIntFloat>{}).
        then(fk::Instantiable<BAddFloat>{5.3});

    static_assert(func2.isSource == false);
    static_assert(func2.is<fk::ReadType>);
    static_assert(func2.params.size == 5);
    using ResType = decltype(func2);
    static_assert(fk::is_fused_operation<typename ResType::Operation>::value);
    using ResOperationTuple = typename ResType::Operation::ParamsType;
    constexpr bool noIntermediateFusedOperation =
        fk::and_v<!fk::is_fused_operation<ResOperationTuple::Operation>::value,
        !fk::is_fused_operation<ResOperationTuple::Next::Operation>::value,
        !fk::is_fused_operation<ResOperationTuple::Next::Next::Operation>::value,
        !fk::is_fused_operation<ResOperationTuple::Next::Next::Next::Operation>::value,
        !fk::is_fused_operation<ResOperationTuple::Next::Next::Next::Next::Operation>::value>;
    static_assert(noIntermediateFusedOperation);

    // All Unary
    constexpr auto func = fk::Instantiable<UFloatInt>{}.then(fk::Instantiable<UIntFloat>{}).then(fk::Instantiable<UFloatInt>{});

    using Operations = decltype(func)::Operation::Operations;
    static_assert(Operations::size == 3);
    static_assert(std::is_same_v<fk::TypeAt_t<0,Operations>, UFloatInt>);
    static_assert(std::is_same_v<fk::TypeAt_t<1, Operations>, UIntFloat>);
    static_assert(std::is_same_v<fk::TypeAt_t<2, Operations>, UFloatInt>);
    static_assert(decltype(func)::Operation::exec(5.5f) == 5);

    constexpr auto op = BAddInt::build(45);

    static_assert(op.params == 45);
    static_assert(decltype(op)::Operation::exec(10, op.params) == 55);

    return 0;
}