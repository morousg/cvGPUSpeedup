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
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/algorithms/basic_ops/set.cuh>
#include <fused_kernel/fused_kernel.cuh>

using namespace fk;

// Operation types
// Read
using RPerThrFloat = PerThreadRead<_2D, float>;
// ReadBack
using RBResize = ResizeRead<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR, Instantiable<RPerThrFloat>>;
// Unary
using UIntFloat = Cast<int, float>;
using UFloatInt = Cast<float, int>;
// Binary
using BAddInt = Add<int>;
using BAddFloat = Add<float>;
// Ternary
using TInterpFloat = Interpolate<InterpolationType::INTER_LINEAR, Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = PerThreadWrite<_2D, float>;
// MidWrite
using MWPerThrFloat = FusedOperation<WPerThrFloat, BAddFloat>;

int launch() {
    constexpr Instantiable<RPerThrFloat> func1{};
    func1.then(UFloatInt::build());
    fuseDF(func1, UFloatInt::build());
    constexpr auto func2 =
        func1.then(Instantiable<UFloatInt>{}).
        then(Instantiable<BAddInt>{4}).
        then(Instantiable<UIntFloat>{}).
        then(Instantiable<BAddFloat>{5.3});

    static_assert(func2.is<ReadType>);
    static_assert(func2.params.size == 5);
    using ResType = decltype(func2);
    static_assert(is_fused_operation<typename ResType::Operation>::value);
    using ResOperationTuple = typename ResType::Operation::ParamsType;
    constexpr bool noIntermediateFusedOperation =
        and_v<!is_fused_operation<ResOperationTuple::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Next::Next::Operation>::value>;
    static_assert(noIntermediateFusedOperation);

    // All Unary
    constexpr auto func = Instantiable<UFloatInt>{}.then(Instantiable<UIntFloat>{}).then(Instantiable<UFloatInt>{});

    using Operations = decltype(func)::Operation::Operations;
    static_assert(Operations::size == 3);
    static_assert(std::is_same_v<TypeAt_t<0,Operations>, UFloatInt>);
    static_assert(std::is_same_v<TypeAt_t<1, Operations>, UIntFloat>);
    static_assert(std::is_same_v<TypeAt_t<2, Operations>, UFloatInt>);
    static_assert(decltype(func)::Operation::exec(5.5f) == 5);

    constexpr auto op = BAddInt::build(45);

    static_assert(op.params == 45);
    static_assert(decltype(op)::Operation::exec(10, op.params) == 55);

    const Ptr2D<uint3> outputAlt(32, 32);
    Ptr2D<uint3> h_output(32, 32, 0, HostPinned);
    constexpr RawPtr<_2D, uchar3> input{nullptr, PtrDims<_2D>(128, 128)};
    constexpr RawPtr<_2D, float3> output{ nullptr, PtrDims<_2D>(32, 32) };
    constexpr Size dstSize(32, 32);
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    constexpr auto someReadOp =
        PerThreadRead<_2D, uchar3>::build(input).then(Cast<uchar3, float3>::build()).then(ResizeRead<INTER_LINEAR>::build(dstSize));

    constexpr OperationTuple<PerThreadRead<_2D, uchar3>, Cast<uchar3, float3>> backFunction = someReadOp.back_function.params;

    constexpr bool correct =
        std::is_same_v<OperationTuple<PerThreadRead<_2D, uchar3>, Cast<uchar3, float3>>, decltype(someReadOp.back_function.params)>;
    static_assert(correct, "Unexpected resulting type");

    constexpr auto finalOp = someReadOp.then(Mul<float3>::build({ { 3.f, 1.f, 32.f } }));

    constexpr auto inputAlt = ReadSet<uchar3>::build({ { { 0,0,0 }, {128,128,1} } });

    constexpr ActiveThreads activeThreads = decltype(inputAlt)::getActiveThreads(inputAlt);

    static_assert(activeThreads.x == 128, "Incorrect size in x");
    static_assert(activeThreads.y == 128, "Incorrect size in y");
    static_assert(activeThreads.z == 1, "Incorrect size in z");

    constexpr auto someReadOpAlt =
        ReadSet<uchar3>::build({ { { 0,0,0 }, {128,128,1} } })
        .then(Cast<uchar3, float3>::build())
        .then(ResizeRead<INTER_LINEAR>::build(dstSize))
        .then(Add<float3>::build({ { 3.f, 1.f, 32.f } }))
        .then(Cast<float3, uint3>::build());

    using ResultingType = decltype(someReadOpAlt);

    constexpr auto res_x = ResultingType::getActiveThreads(someReadOpAlt);

    static_assert(decltype(someReadOpAlt)::getActiveThreads(someReadOpAlt).x == 32, "Wrong width");
    static_assert(decltype(someReadOpAlt)::getActiveThreads(someReadOpAlt).y == 32, "Wrong height");
    static_assert(decltype(someReadOpAlt)::getActiveThreads(someReadOpAlt).z == 1, "Wrong depth");

    executeOperations(stream, someReadOpAlt, PerThreadWrite<_2D, uint3>::build({ outputAlt }));

    gpuErrchk(cudaMemcpy2DAsync(h_output.ptr().data, h_output.dims().pitch,
                                outputAlt.ptr().data, outputAlt.dims().pitch,
                                outputAlt.dims().width * sizeof(uint3),
                                outputAlt.dims().height, cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct2{ true };

    for (uint y = 0; y < 32; ++y) {
        for (uint x = 0; x < 32; ++x) {
            const uint3 temp = *PtrAccessor<_2D>::cr_point({x, y}, h_output.ptr());
            correct2 &= (temp.x == 3 && temp.y == 1 && temp.z == 32);
        }
    }

    return correct2 ? 0 : -1;
}