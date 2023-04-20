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
        const T temp = Operation::exec(x, y, op.ptr.data, 
        op.ptr.dims.width, op.ptr.dims.height, op.ptr.dims.pitch, op.fx, op.fy);
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

template <typename I>
struct interpolate_read_v3 {
    static __device__ __forceinline__ I exec(const uint& x, const uint& y, const I*__restrict__ i_data,
                                             const uint& width, const uint& height, const uint& pitch,
                                             const float& fx, const float& fy) {
        const float src_x = x * fx;
        const float src_y = y * fy;

        const uint x1 = __float2int_rd(src_x);
        const uint y1 = __float2int_rd(src_y);
        const uint x2 = x1 + 1;
        const uint y2 = y1 + 1;        
        const uint x2_read = ::min(x2, width - 1);
        const uint y2_read = ::min(y2, height - 1);

        using floatcn_t = typename VectorType<float, VectorTraits<I>::cn>::type;
        floatcn_t out = make_set<floatcn_t>(0.f);
        uchar3 src_reg = *PtrAccessor<_2D>::cr_point(x1, y1, i_data, pitch);  //input.at_cr({x1, y1});
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y1, i_data, pitch); //*input.at_cr({x2_read, y1});
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x1, y2_read, i_data, pitch); //*input.at_cr({x1, y2_read});
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y2_read, i_data, pitch); //*input.at_cr({x2_read, y2_read});
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        return fk::saturate_cast<I>(out);    
    } 
};

template <typename I>
__device__ __forceinline__ const I exec(const uint& x, const uint& y, const I*__restrict__ i_data,
                                        const uint& width, const uint& height, const uint& pitch,
                                        const float& fx, const float& fy) {
    const float src_x = x * fx;
    const float src_y = y * fy;

    const uint x1 = __float2int_rd(src_x);
    const uint y1 = __float2int_rd(src_y);
    const uint x2 = x1 + 1;
    const uint y2 = y1 + 1;        
    const uint x2_read = ::min(x2, width - 1);
    const uint y2_read = ::min(y2, height - 1);

    using floatcn_t = typename VectorType<float, VectorTraits<I>::cn>::type;
    floatcn_t out = make_set<floatcn_t>(0.f);
    uchar3 src_reg = *PtrAccessor<_2D>::cr_point(x1, y1, i_data, pitch);  //input.at_cr({x1, y1});
    out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

    src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y1, i_data, pitch); //*input.at_cr({x2_read, y1});
    out = out + src_reg * ((src_x - x1) * (y2 - src_y));

    src_reg = *PtrAccessor<_2D>::cr_point(x1, y2_read, i_data, pitch); //*input.at_cr({x1, y2_read});
    out = out + src_reg * ((x2 - src_x) * (src_y - y1));

    src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y2_read, i_data, pitch); //*input.at_cr({x2_read, y2_read});
    out = out + src_reg * ((src_x - x1) * (src_y - y1));

    return fk::saturate_cast<I>(out);    

}

template <typename I>
__global__ void resize_2D_v3(const I*__restrict__ i_data, I* o_data, 
                          uint width, uint height, uint pitch, // Source info
                          uint t_width, uint t_height, uint t_pitch, // Destination info
                          float fx, float fy) {
    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    if ( x < t_width && y < t_height ) {
        I* output = PtrAccessor<_2D>::point(x, y, o_data, t_pitch);
        *output = fk::saturate_cast<I>(interpolate_read_v3<I>::exec(x, y, i_data, width, height, pitch, fx, fy));
    }
}

template <typename I>
__global__ void resize_2D_v2(const I*__restrict__ i_data, I* o_data, 
                          uint width, uint height, uint pitch, // Source info
                          uint t_width, uint t_height, uint t_pitch, // Destination info
                          float fx, float fy) {
    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    if ( x < t_width && y < t_height ) {
        I* output = PtrAccessor<_2D>::point(x, y, o_data, t_pitch);
        *output = fk::saturate_cast<I>(exec(x, y, i_data, width, height, pitch, fx, fy));
    }
}

template <typename I>
__global__ void resize_2D(const I*__restrict__ i_data, I* o_data, 
                          uint width, uint height, uint pitch, // Source info
                          uint t_width, uint t_height, uint t_pitch, // Destination info
                          float fx, float fy) {
    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    if ( x < t_width && y < t_height ) {
        const float src_x = x * fx;
        const float src_y = y * fy;

        const uint x1 = __float2int_rd(src_x);
        const uint y1 = __float2int_rd(src_y);
        const uint x2 = x1 + 1;
        const uint y2 = y1 + 1;        
        const uint x2_read = ::min(x2, width - 1);
        const uint y2_read = ::min(y2, height - 1);

        using floatcn_t = typename VectorType<float, VectorTraits<I>::cn>::type;
        floatcn_t out = make_set<floatcn_t>(0.f);
        uchar3 src_reg = *PtrAccessor<_2D>::cr_point(x1, y1, i_data, pitch);  //input.at_cr({x1, y1});
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y1, i_data, pitch); //*input.at_cr({x2_read, y1});
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *PtrAccessor<_2D>::cr_point(x1, y2_read, i_data, pitch); //*input.at_cr({x1, y2_read});
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *PtrAccessor<_2D>::cr_point(x2_read, y2_read, i_data, pitch); //*input.at_cr({x2_read, y2_read});
        out = out + src_reg * ((src_x - x1) * (src_y - y1));
        
        I* output = PtrAccessor<_2D>::point(x, y, o_data, t_pitch);
        *output = fk::saturate_cast<I>(out);
    }
}

}