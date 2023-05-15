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

#include "operations.cuh"
#include "memory_operations.cuh"

namespace fk { // namespace Fast Kernel

// START OF THE KERNEL RECURSIVITY

// TODO: adapt this to RawPtr<T> instead of raw pointer
/*template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ __forceinline__ void operate_noret(I i_data, binary_operation_pointer<Operation, I, I2, O> op, operations... ops) {
    // we want to have access to I2 in order to ask for the type size for optimizing
    O temp = op.nv_operator(i_data, op.pointer[GLOBAL_ID]);
    operate_noret(temp, ops...);
}*/

// generic operation struct

template <typename Operation>
struct ReadDeviceFunction {
    typename Operation::ParamsType params;
};

template <typename Operation>
struct BinaryDeviceFunction {
    typename Operation::ParamsType params;
};

template <typename Operation>
struct UnaryDeviceFunction {};

template <typename Operation>
struct WriteDeviceFunction {
    typename Operation::ParamsType params;
    dim3 dataDims;
    using Op = Operation;
};

// Util to get the last parameter of a parameter pack
template <typename T>
__device__ __forceinline__ constexpr T last(const T& t) {
    return t;
}
template <typename T, typename... Args>
__device__ __forceinline__ constexpr auto last(const T& t, const Args&... args) {
    return last(args...);
}

// Recursive operate function
template <typename T>
__device__ __forceinline__ constexpr void operate(const Point& thread, const T& i_data) {
    return i_data;
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data, const BinaryDeviceFunction<Operation>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data, op.params), ops...);
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data, const UnaryDeviceFunction<Operation>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data), ops...);
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::Type& i_data, const WriteDeviceFunction<Operation>& op) {
    return i_data;
}

template <typename ReadOperation, typename... operations>
__global__ void cuda_transform(const dim3 gridActiveThreads, const ReadDeviceFunction<ReadOperation> readPattern, const operations... ops) {
    auto writePattern = last(ops...);
    using WriteOperation = typename decltype(writePattern)::Op;

    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
    const Point thread{x, y, z};

    if (x < gridActiveThreads.x && y < gridActiveThreads.y && z < gridActiveThreads.z) {
        const auto tempI = ReadOperation::exec(thread, readPattern.params);
        if constexpr (sizeof...(ops) > 1) {
            const auto tempO = operate(thread, tempI, ops...);
            WriteOperation::exec(thread, tempO, writePattern.params);
        } else {
            WriteOperation::exec(thread, tempI, writePattern.params);
        }
    }
}

void test() {

    RawPtr<_2D, uchar3> input;
    RawPtr<_2D, float3> output;

    auto op1 = ReadDeviceFunction<PerThreadRead<_2D, uchar3>>   { input };
    auto op2 = UnaryDeviceFunction<UnaryCast<uchar3, float3>>   {};
    auto op3 = BinaryDeviceFunction<BinarySum<float3>>          { make_set<float3>(1.f) };
    auto op4 = WriteDeviceFunction<PerThreadWrite<_2D, float3>> { output };

    dim3 gridActiveThreads{1, 1, 1};
    cuda_transform<<<1,1,0,0>>>(gridActiveThreads, op1, op2, op3, op4);

}

}