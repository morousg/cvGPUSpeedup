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

#include <fused_kernel/core/execution_model/fused_operation.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cast.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>

int launch() {
    constexpr auto opTuple1 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>>();

    using OpTuple1Type = decltype(opTuple1);

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");
    static_assert(fk::isUnaryType<typename OpTuple1Type::Operation>, "Wrong Operation Type");

    constexpr auto opTuple2 =
        fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>, fk::Add<int>>
        (fk::OperationData<fk::Add<int>>{3});

    using OpTuple2Type = decltype(opTuple2);

    constexpr fk::DF_t<fk::Add<int, int, int, fk::UnaryType>, fk::Add<int>> df2(3);
    static_assert(df2.params.next.instance.params == 3, "");

    constexpr auto result1 = decltype(df2)::Operation::exec(fk::Tuple<int, int>{4, 4}, df2.params);

    static_assert(result1 == 11, "Wrong result1");

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(fk::isBinaryType<typename OpTuple2Type::Next::Operation>, "Wrong Operation Type");
    static_assert(opTuple2.next.instance.params == 3, "Wrong value");

    constexpr auto opTuple3 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>,
    fk::Cast<int, float>, fk::Cast<float, int>>();

    using OpTuple3Type = decltype(opTuple3);

    constexpr fk::DF_t<fk::Add<int, int, int, fk::UnaryType>,
        fk::Cast<int, float>, fk::Cast<float, int>> df3{};

    constexpr auto result3 = decltype(df3)::Operation::exec(fk::Tuple<int, int>{5,20});
    static_assert(result3 == 25, "Wrong result3");

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; must not compile
    static_assert(fk::isUnaryType<typename OpTuple3Type::Operation>, "Wrong Operation Type");

    return 0;
}