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

#include "operation_parameters.cuh"
#include "memory_operation_parameters.cuh"

namespace fk { // namespace Fast Kernel

// HELPER FUNCTIONS

template <ND D, typename Operation, typename T, typename Enabler = void>
struct split_helper {};

template <ND D, typename Operation, typename T>
struct split_helper<D, Operation, T, typename std::enable_if_t<CN(T) == 2>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& i_data, const split_write_scalar<D, Operation, T>& op) {
        Operation::exec(thread, i_data, op.x, op.y);
    }
};

template <ND D, typename Operation, typename T>
struct split_helper<D, Operation, T, typename std::enable_if_t<CN(T) == 3>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& i_data, const split_write_scalar<D, Operation, T>& op) {
        Operation::exec(thread, i_data, op.x, op.y, op.z);
    }
};

template <ND D, typename Operation, typename T>
struct split_helper<D, Operation, T, typename std::enable_if_t<CN(T) == 4>> {
    FK_DEVICE_FUSE void exec(const Point& thread, const T& i_data, const split_write_scalar<D, Operation, T>& op) {
        Operation::exec(thread, i_data, op.x, op.y, op.z, op.w);
    }
};

// START OF THE KERNEL RECURSIVITY

template <typename T>
__device__ __forceinline__ constexpr void operate_noret(const Point& thread, const T& i_data) {}

template <ND D, typename Operation, typename T, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const Point& thread, const T& i_data, const split_write_scalar<D, Operation, T>& op, const operations&... ops) {
    split_helper<D, Operation, T>::exec(thread, i_data, op);
    operate_noret(thread, i_data, ops...);
}

template <typename Operation, typename T, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const Point& thread, const T& i_data, const split_write_tensor<Operation, typename fk::VectorTraits<T>::base>& op, const operations&... ops) {
    Operation::exec(thread, i_data, op.t);
    operate_noret(thread, i_data, ops...);
}

template <ND D, typename Operation, typename T, typename... operations>
__device__ __forceinline__ constexpr 
void operate_noret(const Point& thread, const T& i_data, const memory_write_scalar<D, Operation, T>& op, const operations&... ops) {
    Operation::exec(thread, i_data, op.ptr);
    operate_noret(thread, i_data, ops...);
}

template <typename I, typename O, typename Operation, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const Point& thread, const I& i_data, const unary_operation_scalar<Operation, O>& op, const operations&... ops) {
    const O temp = Operation::exec(i_data);
    operate_noret(thread, temp, ops...);
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const Point& thread, const I& i_data, const binary_operation_scalar<Operation, I2, O>& op, const operations&... ops) {
    const O temp = Operation::exec(i_data, op.scalar);
    operate_noret(thread, temp, ops...);
}

// TODO: adapt this to RawPtr<T> instead of raw pointer
/*template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ __forceinline__ void operate_noret(I i_data, binary_operation_pointer<Operation, I, I2, O> op, operations... ops) {
    // we want to have access to I2 in order to ask for the type size for optimizing
    O temp = op.nv_operator(i_data, op.pointer[GLOBAL_ID]);
    operate_noret(temp, ops...);
}*/

template <int NPtr, typename Operation, typename T, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const memory_read_iterpolated_N<NPtr, Operation, T>& op, const operations&... ops) {
    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes

    if (x < op.target_width && y < op.target_height && z < op.active_planes) {
        const T temp = Operation::exec(Point(x,y,z), op.ptr, op.fx, op.fy);
        operate_noret(Point(x, y, z), temp, ops...);
    }
}

template<ND D, typename I, typename... operations>
__global__ void cuda_transform_(const RawPtr<D, I> i_data, const operations... ops) {
    cg::thread_block g =  cg::this_thread_block();
    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes

    if (x < i_data.dims.width && y < i_data.dims.height) {
        operate_noret(Point(x, y, z), *PtrAccessor<D>::cr_point(Point(x, y, z), i_data), ops...);
    }
}

template<typename... operations>
__global__ void cuda_transform_noret_2D(const operations... ops) {
    operate_noret(ops...);
}

// generic operation struct

template <typename Params, typename Operation, typename I>
struct ReadDeviceFunction {
    Params params;
};

template <typename Params, typename Operation, typename I, typename O>
struct BinaryDeviceFunction {
    Params params;
};

template <typename Operation, typename I, typename O>
struct UnaryDeviceFunction {};

template <typename Params, typename Operation, typename O>
struct WriteDeviceFunction {
    Params params;
    dim3 dataDims;
    Operation write;
    using type = O;
};

// Util to get the last parameter of a parameter pack
template<typename... Args>
__device__ __forceinline__ constexpr decltype(auto) last(Args&&... args){
   return (std::forward<Args>(args), ...);
}

// Recursive operate function
template <typename T>
__device__ __forceinline__ constexpr void operate(const Point& thread, const T& i_data) {
    return i_data;
}

template <typename I, typename O, typename Params, typename Operation, typename... operations>
__device__ __forceinline__ constexpr O operate(const Point& thread, const I& i_data, const BinaryDeviceFunction<Params, Operation, I, O>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data, op.params), ops...);
}

template <typename I, typename O, typename Operation, typename... operations>
__device__ __forceinline__ constexpr O operate(const Point& thread, const I& i_data, const UnaryDeviceFunction<Operation, I, O>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data), ops...);
}

template <typename O, typename Params, typename Operation, typename... operations>
__device__ __forceinline__ constexpr O operate(const Point& thread, const O& i_data, const WriteDeviceFunction<Params, Operation, O>& op) {
    return i_data;
}

template <typename ReadParams, typename ReadOperation, typename I, typename... operations>
__global__ void cuda_transform(const ReadDeviceFunction<ReadParams, ReadOperation, I> readPattern, 
                                                  const operations&... ops) {
    auto writePattern = last(ops...);
    using O = typename decltype(writePattern)::type;
    using WriteOperation = decltype(writePattern.write);

    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
    const Point thread{x, y, z};

    if (x < writePattern.dataDims.x && y < writePattern.dataDims.y && z < writePattern.dataDims.z) {
        const I tempI = ReadOperation::exec(thread, readPattern.params);
        if constexpr (sizeof...(ops) > 1) {
            const O tempO = operate(thread, tempI, ops...);
            WriteOperation::exec(thread, tempO, writePattern.params);
        } else {
            WriteOperation::exec(thread, tempI, writePattern.params);
        }
    }
}

void test() {

    RawPtr<_2D, uchar3> input;
    RawPtr<_2D, float3> output;

    auto op1 = ReadDeviceFunction<RawPtr<_2D, uchar3>, perthread_read<_2D, uchar3>, uchar3>{input};
    auto op2 = UnaryDeviceFunction<unary_cast<uchar3,float3>, uchar3, float3>{};
    auto op3 = WriteDeviceFunction<RawPtr<_2D, float3>, perthread_write<_2D, float3>, float3>{output, {1, 1, 1}};

    cuda_transform<<<1,1,0,0>>>(op1, op2, op3);

}

}