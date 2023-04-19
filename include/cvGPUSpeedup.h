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

#include <external/carotene/saturate_cast.hpp>
#include <cvGPUSpeedupHelpers.h>
#include <fast_kernel/fast_kernel.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvGS {

template <int I, int O>
inline constexpr fk::unary_operation_scalar<fk::unary_cast<CUDA_T(I), CUDA_T(O)>, CUDA_T(I), CUDA_T(O)> convertTo() {
    return {};
}

template <int I>
inline constexpr fk::binary_operation_scalar<fk::binary_mul<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> multiply(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::binary_operation_scalar<fk::binary_mul<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>::build(src2);
}

template <int I>
inline constexpr fk::binary_operation_scalar<fk::binary_sub<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> subtract(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::binary_operation_scalar<fk::binary_sub<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>::build(src2);
}

template <int I>
inline constexpr fk::binary_operation_scalar<fk::binary_div<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> divide(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::binary_operation_scalar<fk::binary_div<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>::build(src2);
}

template <int I>
inline constexpr fk::binary_operation_scalar<fk::binary_sum<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> add(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::binary_operation_scalar<fk::binary_sum<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>::build(src2);
}

template <int I>
inline constexpr fk::split_write_scalar_2D<fk::perthread_split_write_2D<CUDA_T(I)>, CUDA_T(I)> split(const std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr3D<BASE_CUDA_T(I)>> fk_output;
    for (auto& mat : output) {
        fk::Ptr3D<BASE_CUDA_T(I)> o_ptr((BASE_CUDA_T(I)*)mat.data, mat.cols, mat.rows, mat.step);
        fk_output.push_back(o_ptr);
    }
    return internal::split_builder_t<I, fk::split_write_scalar_2D<fk::perthread_split_write_2D<CUDA_T(I)>, CUDA_T(I)>>::build(fk_output);
}

template <int T, int INTER_F>
inline constexpr fk::memory_read_iterpolated<fk::interpolate_read<CUDA_T(T), (fk::InterpolationType)INTER_F>, CUDA_T(T)> resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    // So far we only support fk::INTER_LINEAR
    const fk::Ptr3D<CUDA_T(T)> fk_input((CUDA_T(T)*)input.data, input.cols, input.rows, input.step);

    uint t_width, t_height;
    if (dsize != cv::Size()) {
        fx = static_cast<double>(dsize.width) / fk_input.width();
        fy = static_cast<double>(dsize.height) / fk_input.height();
        t_width = dsize.width;
        t_height = dsize.height;
    } else {
        t_width = CAROTENE_NS::internal::saturate_cast<int>(fk_input.width() * fx);
        t_height = CAROTENE_NS::internal::saturate_cast<int>(fk_input.height() * fy);
    }

    return { fk_input, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy), t_width, t_height };
}

template <int O>
inline constexpr fk::memory_write_scalar_2D<fk::perthread_write_2D<CUDA_T(O)>, CUDA_T(O)> write(const cv::cuda::GpuMat& output) {
    fk::Ptr3D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);
    return { fk_output };
}

template <int T, typename... operations>
inline const dim3 extractDataDims(const fk::memory_read_iterpolated<fk::interpolate_read<CUDA_T(T), fk::InterpolationType::INTER_LINEAR>, CUDA_T(T)>& op, const operations&... ops) {
    return dim3(op.target_width, op.target_height, op.ptr.planes);
}

template <int T, typename... operations>
inline constexpr void executeOperations(cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    dim3 dataDims = extractDataDims<T>(ops...);
    dim3 block = fk::getBlockSize(dataDims.x, dataDims.y);
    dim3 grid;
    grid.x = (unsigned int)ceil(dataDims.x / (float)block.x);
    grid.y = (unsigned int)ceil(dataDims.y / (float)block.y);
    fk::cuda_transform_noret_2D<<<grid, block, 0, cu_stream>>>(ops...);

    gpuErrchk(cudaGetLastError());
}

template <int I, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr3D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid;
    grid.x = (unsigned int)ceil(fk_input.width() / (float)block.x);
    grid.y = (unsigned int)ceil(fk_input.height() / (float)block.y);
    fk::cuda_transform_<<<grid, block, 0, cu_stream>>>(fk_input.d_ptr(), ops...);

    gpuErrchk(cudaGetLastError());
}

template <int I, int O, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr3D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);
    fk::Ptr3D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid(ceil(fk_input.width() / (float)block.x), ceil(fk_input.height() / (float)block.y));

    fk::memory_write_scalar_2D<fk::perthread_write_2D<CUDA_T(O)>, CUDA_T(O)> opFinal = { fk_output };
    fk::cuda_transform_<<<grid, block, 0, cu_stream>>>(fk_input.d_ptr(), ops..., opFinal);
    
    gpuErrchk(cudaGetLastError());
}
} // namespace cvGS
