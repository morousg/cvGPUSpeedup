/* Copyright 2024-2025 Oscar Amoros Huguet

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
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/algorithms/image_processing/color_conversion.cuh>

using namespace fk;

bool test_fuseDFResultingTypes() {

    constexpr Read<PerThreadRead<_2D, float>> readOp{};
    constexpr Binary<Add<float>> addOp{ 3.f };
    constexpr Unary<Cast<float, int>> castOp{};
    constexpr Write<PerThreadWrite<_2D, float>> writeOp{};

    using Test = decltype(PerThreadRead<_2D, float>::num_elems_y(std::declval<Point>(), std::declval<typename PerThreadRead<_2D, float>::ParamsType>()));

    static_assert(std::is_same_v<Test, uint>);

    constexpr auto fused1 = fuseDF(readOp, addOp, castOp);

    constexpr auto read = Read<PerThreadRead<_2D, float>>{ { fk::RawPtr<_2D, float>{nullptr, {128, 4}} } };

    constexpr auto readOp2 = PerThreadRead<_2D, uchar3>::build(RawPtr<_2D, uchar3>{nullptr, PtrDims<_2D>(128,128)});

    constexpr auto readYUV = ReadYUV<PixelFormat::NV12>::build(RawPtr<_2D, uchar>{nullptr, PtrDims<_2D>(128, 128)});
    constexpr auto readRGB = readYUV.then(ConvertYUVToRGB<PixelFormat::NV12, ColorRange::Full, ColorPrimitives::bt2020, false>::build());

    constexpr auto resizeRead = ResizeRead<INTER_LINEAR>::build(readRGB, Size(64, 64)).then(Mul<float>::build(3.f)).then(Div<float>::build(4.3f));

    //decltype(fused1)::

    static_assert(std::is_same_v<typename decltype(fused1)::Operation,
        fk::FusedOperation<fk::PerThreadRead<fk::_2D, float>, fk::Add<float>, fk::Cast<float, int>>>, "Unexpected type after fuseDF");

    constexpr bool result1 = fk::is_fused_operation<fk::FusedOperation<fk::PerThreadRead<fk::_2D, float>, fk::Add<float>, fk::Cast<float, int>>>::value;

    constexpr bool result2 = fk::is_fused_operation<typename decltype(fused1)::Operation>::value;

    static_assert(result1 && result2, "is_fused_operation does not work properly");

    const auto fused2 = fk::fuseDF(readOp, addOp, writeOp);

    return result1 && result2;
}

constexpr bool test_fuseFusedOperations() {
    const fk::Read<fk::PerThreadRead<fk::_2D, float>> readOp{};
    const fk::Binary<fk::Add<float>> addOp{ 3.f };
    const fk::Unary<fk::Cast<float, int>> castOp{};
    const fk::Write<fk::PerThreadWrite<fk::_2D, float>> writeOp{};

    const auto fused1 = fk::fuseDF(readOp, addOp);
    const auto fused2 = fk::fuseDF(fused1, castOp);

    return true;
}

int launch() {
    constexpr auto opTuple1 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>>();

    using OpTuple1Type = decltype(opTuple1);

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");
    static_assert(fk::isUnaryType<typename OpTuple1Type::Operation>, "Wrong Operation Type");

    constexpr auto opTuple2 =
        fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>, fk::Add<int>>
        (fk::OperationData<fk::Add<int>>{3});

    using OpTuple2Type = decltype(opTuple2);

    constexpr auto df2 = fk::Add<int, int, int, fk::UnaryType>::build().then(fk::Add<int >::build(3));
    static_assert(df2.params.next.instance.params == 3, "");

    constexpr auto result1 = decltype(df2)::Operation::exec(fk::Tuple<int, int>{4, 4}, df2.params);

    static_assert(result1 == 11, "Wrong result1");

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(fk::isBinaryType<typename OpTuple2Type::Next::Operation>, "Wrong Operation Type");
    static_assert(opTuple2.next.instance.params == 3, "Wrong value");

    constexpr auto opTuple3 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>,
    fk::Cast<int, float>, fk::Cast<float, int>>();

    using OpTuple3Type = decltype(opTuple3);

    constexpr auto df3 = fk::Add<int, int, int, fk::UnaryType>::build().then(fk::Cast<int, float>::build()).then(fk::Cast<float, int>::build());

    constexpr auto result3 = decltype(df3)::Operation::exec(fk::Tuple<int, int>{5,20});
    static_assert(result3 == 25, "Wrong result3");

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; must not compile
    static_assert(fk::isUnaryType<typename OpTuple3Type::Operation>, "Wrong Operation Type");

    //static_assert(test_fuseDFResultingTypes(), "Something wrong with the types generated by fusedDF");
    static_assert(test_fuseFusedOperations(), "Something wrong while fusing a FusedOperation with another operation");

    return 0;
}