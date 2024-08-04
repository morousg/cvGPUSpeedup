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

#include <tests/fkTestsCommon.h>
#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION_NAME[]{ "Number of Operations" };
constexpr size_t NUM_EXPERIMENTS = 3;
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 100;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimanesionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 1920 * 1080;

template <typename T>
__global__ void init_values(const T val, fk::RawPtr<fk::_1D, T> pointer_to_init) {
    const int x = threadIdx.x + (blockDim.x * blockIdx.x);
    if (x < pointer_to_init.dims.width) {
        *fk::PtrAccessor<fk::_1D>::point(fk::Point(x), pointer_to_init) = val;
    }
}

template <typename InputType, typename OutputType, size_t NumOps, typename DeviceFunction>
struct VerticalFusion {
    static inline void execute(const fk::Ptr1D<InputType>& input, const cudaStream_t& stream,
                               const fk::Ptr1D<OutputType>& output, const DeviceFunction& dFunc) {
        const dim3 activeThreads{ output.ptr().dims.width };
        fk::SourceRead<fk::PerThreadRead<fk::_1D, InputType>> readDF{ input, activeThreads };
        using Loop = fk::Binary<fk::StaticLoop<fk::StaticLoop<typename DeviceFunction::Operation, INCREMENT>, NumOps/INCREMENT>>;
        Loop loop;
        loop.params = dFunc.params;

        dim3 block(256);
        dim3 grid(ceil(NUM_ELEMENTS / (float)block.x));
        fk::cuda_transform<<<grid, block, 0, stream>>>(readDF, loop, fk::Write<fk::PerThreadWrite<fk::_1D, OutputType>>{ output });
    }
};

template <int VARIABLE_DIMENSION>
inline int testLatencyHiding() {

    const fk::Ptr1D<uchar3> input(NUM_ELEMENTS);
    const fk::Ptr1D<uchar3> output(NUM_ELEMENTS);

    constexpr uchar3 init_val{ 1,2,3 };

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    dim3 block(256);
    dim3 grid(ceil(NUM_ELEMENTS / (float)block.x));
    init_values<<<grid, block, 0, stream>>>(init_val, input.ptr());

    fk::Binary<fk::Add<uchar3>> df{ fk::make_set<uchar3>(2) };

    START_CVGS_BENCHMARK

        VerticalFusion<uchar3, uchar3, VARIABLE_DIMENSION, fk::Binary<fk::Add<uchar3>>>::execute(input, stream, output, df);

    STOP_CVGS_BENCHMARK

        return 0;
}

template <int... Idx>
inline int testLatencyHidingHelper(std::integer_sequence<int, Idx...>&) {
    const bool result = ((testLatencyHiding<variableDimanesionValues[Idx]>() == 0) && ...);
    if (result) {
        return 0;
    } else {
        return -1;
    }
}
#endif

int launch() {
#ifdef ENABLE_BENCHMARK
    const int result =  testLatencyHidingHelper(std::make_integer_sequence<int, variableDimanesionValues.size()>{});

    CLOSE_BENCHMARK
    return result;
#else
    return 0;
#endif
}
