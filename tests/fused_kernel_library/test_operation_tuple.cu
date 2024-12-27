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

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/operation_tuple.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cast.cuh>
#include <fused_kernel/algorithms/image_processing/saturate.cuh>

bool test_OTInitialization() {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    constexpr uint X = 64;
    constexpr uint Y = 64;

    const fk::Ptr2D<uchar> input(64, 64);
    constexpr fk::ActiveThreads gridActiveThreads(X, Y);
    using Op = fk::PerThreadRead<fk::_2D, uchar>;
    const fk::SourceRead<Op> read{ {input}, gridActiveThreads };

    const fk::OperationTuple<Op> testing{ {read.params} };

    const auto test2 = fk::devicefunctions_to_operationtuple(read);
    const fk::SourceRead<fk::FusedOperation<Op>> test3 = fk::fuseDF(read);

    using Op2 = fk::SaturateCast<uchar, uint>;
    constexpr fk::Unary<Op2> cast = {};

    const auto ot1 = fk::devicefunctions_to_operationtuple(read);
    constexpr auto ot2 = fk::devicefunctions_to_operationtuple(cast);

    const auto test4 = fk::make_operation_tuple_<Op, Op2>(ot1.instance);
    const auto test5 = fk::make_operation_tuple_<Op, Op2>(fk::get<0>(ot1));

    constexpr auto filtered1 =
        fk::filtered_integer_sequence_t<int, fk::NotUnaryRestriction, fk::TypeList<typename Op::InstanceType>>{};

    const fk::OperationTuple<Op, decltype(ot2)::Operation> test6 = fk::cat(ot1, ot2);

    const auto test7 = fk::devicefunctions_to_operationtuple(read, cast);

    const auto test8 = fk::fuseDF(read, cast);

    const auto test9 = fk::make_source(fk::Instantiable<fk::FusedOperation<typename decltype(read)::Operation,
                                                                   typename decltype(cast)::Operation>>
    { fk::devicefunctions_to_operationtuple(read, cast) });

    return true;
}

int launch() {
    constexpr auto opTuple1 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>>();

    using OpTuple1Type = decltype(opTuple1);

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");
    static_assert(fk::isUnaryType<typename OpTuple1Type::Operation>, "Wrong Operation Type");

    constexpr fk::OperationData<fk::Add<int>> data(3);

    constexpr auto opTuple2 =
        fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>, fk::Add<int>>
        (fk::OperationData<fk::Add<int>>(3));

    using OpTuple2Type = decltype(opTuple2);

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(fk::isBinaryType<typename OpTuple2Type::Next::Operation>, "Wrong Operation Type");
    static_assert(opTuple2.next.instance.params == 3, "Wrong value");

    constexpr auto opTuple3 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>,
    fk::Cast<int, float>, fk::Cast<float, int>>();

    using OpTuple3Type = decltype(opTuple3);

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; must not compile
    static_assert(fk::isUnaryType<typename OpTuple3Type::Operation>, "Wrong Operation Type");
    static_assert(opTuple2.next.instance.params == 3, "Wrong value");

    if (!test_OTInitialization()) {
        return -1;
    }

    return 0;
}