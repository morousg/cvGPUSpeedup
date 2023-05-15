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
#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/ptr_nd.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvGS {

template <int I, int O>
inline constexpr fk::UnaryDeviceFunction<fk::UnaryCast<CUDA_T(I), CUDA_T(O)>> convertTo() {
    return {};
}

template <int I>
inline constexpr fk::BinaryDeviceFunction<fk::BinaryMul<CUDA_T(I)>> multiply(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::BinaryDeviceFunction<fk::BinaryMul<CUDA_T(I)>>>::build(src2);
}

template <int I>
inline constexpr fk::BinaryDeviceFunction<fk::BinarySub<CUDA_T(I)>> subtract(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::BinaryDeviceFunction<fk::BinarySub<CUDA_T(I)>>>::build(src2);
}

template <int I>
inline constexpr fk::BinaryDeviceFunction<fk::BinaryDiv<CUDA_T(I)>> divide(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::BinaryDeviceFunction<fk::BinaryDiv<CUDA_T(I)>>>::build(src2);
}

template <int I>
inline constexpr fk::BinaryDeviceFunction<fk::BinarySum<CUDA_T(I)>> add(const cv::Scalar& src2) {
    return internal::operator_builder_t<I, fk::BinaryDeviceFunction<fk::BinarySum<CUDA_T(I)>>>::build(src2);
}

template <int T, cv::ColorConversionCodes CODE>
inline constexpr auto cvtColor() {
    // So far, we only support reordering channels
    static_assert(isSupportedColorConversion<CODE>, "Color conversion type not supported yet.");
    if constexpr (CODE == cv::COLOR_BGR2RGB || CODE == cv::COLOR_RGB2BGR) {
        return fk::UnaryDeviceFunction<fk::UnaryVectorReorder<CUDA_T(T), 2, 1, 0>> {};
    } else if constexpr (CODE == cv::COLOR_BGRA2RGBA || CODE == cv::COLOR_RGBA2BGRA) {
        return fk::UnaryDeviceFunction<fk::UnaryVectorReorder<CUDA_T(T), 2, 1, 0, 3>> {};
    }
}

template <int O>
inline constexpr auto split(const std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
    for (auto& mat : output) {
        fk::Ptr2D<BASE_CUDA_T(O)> o_ptr((BASE_CUDA_T(O)*)mat.data, mat.cols, mat.rows, mat.step);
        fk_output.push_back(o_ptr);
    }
    return internal::split_builder_t<O, fk::Ptr2D<BASE_CUDA_T(O)>, fk::WriteDeviceFunction<fk::SplitWrite<fk::_2D, CUDA_T(O)>>>::build(fk_output);
}

template <int I>
inline constexpr fk::WriteDeviceFunction<fk::TensorSplitWrite<CUDA_T(I)>> split(const cv::cuda::GpuMat& output, const cv::Size& planeDims) {
    assert(output.cols % (planeDims.width * planeDims.height) == 0 && output.cols / (planeDims.width * planeDims.height) == CV_MAT_CN(I) &&
    "Each row of the GpuMap should contain as many planes as width / (planeDims.width * planeDims.height)");

    fk::Tensor<BASE_CUDA_T(I)> t_output((BASE_CUDA_T(I)*)output.data, planeDims.width, planeDims.height, output.rows, CV_MAT_CN(I));

    return {t_output};
}

template <int T, int INTER_F>
inline const fk::ReadDeviceFunction<fk::InterpolateRead<CUDA_T(T), (fk::InterpolationType)INTER_F>>
    resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const fk::RawPtr<fk::_2D, CUDA_T(T)> fk_input = 
    {(CUDA_T(T)*)input.data, {(uint)input.cols, (uint)input.rows, (uint)input.step}};

    if (dsize != cv::Size()) {
        fx = static_cast<double>(dsize.width) / input.cols;
        fy = static_cast<double>(dsize.height) / input.rows;
        return {{fk_input, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy)}, {(uint)dsize.width, (uint)dsize.height}};
    } else {
        return {{fk_input, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy)},
                 {CAROTENE_NS::internal::saturate_cast<uint>(input.cols * fx),
                  CAROTENE_NS::internal::saturate_cast<uint>(input.rows * fy)}};
    }
}

template <int T, int INTER_F, int NPtr>
inline const auto resize(const std::array<cv::cuda::GpuMat, NPtr>& input, const cv::Size& dsize, const int usedPlanes) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    fk::ReadDeviceFunction<fk::BatchRead<fk::InterpolateRead<CUDA_T(T), (fk::InterpolationType)INTER_F>, NPtr>> resizeArray;
    resizeArray.activeThreads.x = dsize.width;
    resizeArray.activeThreads.y = dsize.height;
    resizeArray.activeThreads.z = usedPlanes;

    for (int i=0; i<usedPlanes; i++) {
        // So far we only support fk::INTER_LINEAR
        fk::PtrDims<fk::_2D> dims;
        dims.width = (uint)input[i].cols;
        dims.height = (uint)input[i].rows;
        dims.pitch = (uint)input[i].step;
        resizeArray.params[i].ptr = {(CUDA_T(T)*)input[i].data, dims};
        resizeArray.params[i].fx = static_cast<float>(1.0 / (static_cast<double>(dsize.width) / input[i].cols));
        resizeArray.params[i].fy = static_cast<float>(1.0 / (static_cast<double>(dsize.height) / input[i].rows));
    }

    return resizeArray;
}

template <int O>
inline constexpr fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>> write(const cv::cuda::GpuMat& output) {
    fk::Ptr2D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);
    return { fk_output };
}

template <typename Operation, typename... operations>
inline dim3 extractDataDims(const fk::ReadDeviceFunction<Operation>& op, const operations&... ops) {
    return op.activeThreads;
}

template <typename... operations>
inline constexpr void executeOperations(cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    dim3 dataDims = extractDataDims(ops...);
    dim3 block = fk::getBlockSize(dataDims.x, dataDims.y);
    dim3 grid;
    grid.x = (unsigned int)ceil(dataDims.x / (float)block.x);
    grid.y = (unsigned int)ceil(dataDims.y / (float)block.y);
    grid.z = dataDims.z;

    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(ops...);

    gpuErrchk(cudaGetLastError());
}

template <int I, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid;
    grid.x = (unsigned int)ceil(fk_input.dims().width / (float)block.x);
    grid.y = (unsigned int)ceil(fk_input.dims().height / (float)block.y);
    dim3 gridActiveThreads(fk_input.dims.width, fk_input.dims.height);

    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, CUDA_T(I)>>{fk_input, gridActiveThreads}, ops...);

    gpuErrchk(cudaGetLastError());
}

template <int I, int O, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, const operations&... ops) {
    cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);
    fk::Ptr2D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);

    dim3 block = fk_input.getBlockSize();
    dim3 grid(ceil(fk_input.dims().width / (float)block.x), ceil(fk_input.dims().height / (float)block.y));
    dim3 gridActiveThreads(fk_input.dims().width, fk_input.dims().height);
    fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> firstOp { fk_input, gridActiveThreads };
    fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>> opFinal { fk_output };
    
    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(firstOp, ops..., opFinal);
    
    gpuErrchk(cudaGetLastError());
}
} // namespace cvGS
