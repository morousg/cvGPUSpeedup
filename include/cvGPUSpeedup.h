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

#include "cv2cuda_types.h"
#include "fast_kernel.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

// Oscar: this is a prototype interface to my kernel fusion library, to test
// OpenCV programmer's opinion on the shape of it.

namespace cvGS {

template <int I, int O>
unary_operation_scalar<unary_cuda_vector_cast<CUDA_T(I), CUDA_T(O)>, CUDA_T(I), CUDA_T(O)> convertTo() {
    return {};
}

// This are just a quick mockup of the future generic functions
// They only work with types that have 3 components. In the future
// they will work with anything.
template <int I>
binary_operation_scalar<binary_mul<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> multiply(cv::Scalar src2) {
    return {make_<CUDA_T(I)>(src2[0], src2[1], src2[2])};
}

template <int I>
binary_operation_scalar<binary_sub<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> subtract(cv::Scalar src2) {
    return {make_<CUDA_T(I)>(src2[0], src2[1], src2[2])};
}

template <int I>
binary_operation_scalar<binary_div<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> divide(cv::Scalar src2) {
    return {make_<CUDA_T(I)>(src2[0], src2[1], src2[2])};
}

template <int I, typename Operator, typename Enabler = void>
struct split_t {};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CHANNELS(I) == 2>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data};
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CHANNELS(I) == 3>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data,
                (BASE_CUDA_T(I)*)output.at(2).data};
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CHANNELS(I) == 4>> {
    inline constexpr Operator operator()(std::vector<cv::cuda::GpuMat>& output) {
        return {(BASE_CUDA_T(I)*)output.at(0).data, (BASE_CUDA_T(I)*)output.at(1).data,
                (BASE_CUDA_T(I)*)output.at(2).data, (BASE_CUDA_T(I)*)output.at(3).data};
    }
};

template <int I>
split_write_scalar<perthread_split_write<CUDA_T(I)>, CUDA_T(I)> split(std::vector<cv::cuda::GpuMat>& output) {
    return split_t<I, split_write_scalar<perthread_split_write<CUDA_T(I)>, CUDA_T(I)>>()(output);
}

template <int I, typename... operations>
void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, operations... ops) {
    int num_elems = input.rows * input.cols;

    dim3 block(256);
    dim3 grid(ceil(num_elems / (float)block.x));
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    cuda_transform_noret<<<grid, block, 0, cu_stream>>>(num_elems, (CUDA_T(I)*)input.data, ops...);
    gpuErrchk(cudaGetLastError());
}

template <int I, int O, typename... operations>
void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, operations... ops) {
    int num_elems = input.rows * input.cols;

    dim3 block(256);
    dim3 grid(ceil(num_elems / (float)block.x));
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    memory_write_scalar<perthread_write<CUDA_T(O)>, CUDA_T(O), CUDA_T(O)> opFinal = { (CUDA_T(O)*)output.data };

    cuda_transform_noret<<<grid, block, 0, cu_stream>>>(num_elems, (CUDA_T(I)*)input.data, ops..., opFinal);
    gpuErrchk(cudaGetLastError());
}

} // namespace cvGS
