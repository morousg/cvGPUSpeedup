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

#include <cvGPUSpeedupHelpers.h>
#include <fast_kernel/fast_kernel.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvGS {

template <int I, int O>
fk::unary_operation_scalar<fk::unary_cuda_vector_cast<CUDA_T(I), CUDA_T(O)>, CUDA_T(I), CUDA_T(O)> convertTo() {
    return {};
}

template <int I>
fk::binary_operation_scalar<fk::binary_mul<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> multiply(cv::Scalar src2) {
    return internal::operate_t<I, fk::binary_operation_scalar<fk::binary_mul<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>()(src2);
}

template <int I>
fk::binary_operation_scalar<fk::binary_sub<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> subtract(cv::Scalar src2) {
    return internal::operate_t<I, fk::binary_operation_scalar<fk::binary_sub<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>()(src2);
}

template <int I>
fk::binary_operation_scalar<fk::binary_div<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> divide(cv::Scalar src2) {
    return internal::operate_t<I, fk::binary_operation_scalar<fk::binary_div<CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>()(src2);
}

template <int I>
fk::binary_operation_scalar<fk::binary_sum<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)> add(cv::Scalar src2) {
    return internal::operate_t<I, fk::binary_operation_scalar<fk::binary_sum<CUDA_T(I), CUDA_T(I)>, CUDA_T(I), CUDA_T(I)>>()(src2);
}

template <int I>
fk::split_write_scalar_2D<fk::perthread_split_write_2D<CUDA_T(I)>, CUDA_T(I)> split(std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr3D<BASE_CUDA_T(I)>> fk_output;
    for (auto& mat : output) {
        fk::Ptr3D<BASE_CUDA_T(I)> o((BASE_CUDA_T(I)*)mat.data, mat.cols, mat.rows, mat.step);
        fk_output.push_back(o);
    }
    return internal::split_t<I, fk::split_write_scalar_2D<fk::perthread_split_write_2D<CUDA_T(I)>, CUDA_T(I)>>()(fk_output);
}

template <int I, typename... operations>
void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, operations... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr3D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid;
    grid.x = (unsigned int)ceil(fk_input.width() / (float)block.x);
    grid.y = (unsigned int)ceil(fk_input.height() / (float)block.y);
    fk::cuda_transform_noret_2D<<<grid, block, 0, cu_stream>>>(fk_input.d_ptr(), ops...);

    gpuErrchk(cudaGetLastError());
}

template <int I, int O, typename... operations>
void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, operations... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr3D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);
    fk::Ptr3D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid(ceil(fk_input.width() / (float)block.x), ceil(fk_input.height() / (float)block.y));

    fk::memory_write_scalar_2D<fk::perthread_write_2D<CUDA_T(O)>, CUDA_T(O)> opFinal = { fk_output };
    fk::cuda_transform_noret_2D<<<grid, block, 0, cu_stream>>>(fk_input.d_ptr(), ops..., opFinal);
    
    gpuErrchk(cudaGetLastError());
}

} // namespace cvGS
