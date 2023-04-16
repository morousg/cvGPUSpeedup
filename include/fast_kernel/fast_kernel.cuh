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

#include "operation_patterns.cuh"
#include "memory_operation_patterns.cuh"

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

template <typename Operation, typename T, typename... operations>
__device__ __forceinline__ constexpr void operate_noret(const memory_read_iterpolated_N<1, Operation, T>& op, const operations&... ops) {
    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes

    if (x < op.target_width && y < op.target_height) {
        const T temp = Operation::exec({x, y, z}, op.ptr[z].data, op.ptr[z].dims, op.fy[z], op.fx[z]);
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
        operate_noret(Point(x, y, z), *PtrAccessor<D>::cr_point({x, y, z}, i_data.data, i_data.dims), ops...);
    }
}

template<typename... operations>
__global__ void cuda_transform_noret_2D(const operations... ops) {
    operate_noret(ops...);
}

}