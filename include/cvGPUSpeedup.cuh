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
#include <cvGPUSpeedupHelpers.cuh>
#include <fused_kernel/fused_kernel_launchers.cuh>
#include <fused_kernel/device_function_builders.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>

namespace cvGS {

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1 };

template <int I, int O>
inline constexpr auto convertTo() {
    return fk::UnaryDeviceFunction<fk::UnaryCast<CUDA_T(I), CUDA_T(O)>>{};
}

template <int I>
inline constexpr auto multiply(const cv::Scalar& src2) {
    return fk::BinaryDeviceFunction<fk::BinaryMul<CUDA_T(I)>> { internal::cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto subtract(const cv::Scalar& src2) {

    return fk::BinaryDeviceFunction<fk::BinarySub<CUDA_T(I)>> { internal::cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto divide(const cv::Scalar& src2) {
    return fk::BinaryDeviceFunction<fk::BinaryDiv<CUDA_T(I)>> { internal::cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto add(const cv::Scalar& src2) {

    return fk::BinaryDeviceFunction<fk::BinarySum<CUDA_T(I)>> { internal::cvScalar2CUDAV<I>::get(src2) };
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
        const fk::Ptr2D<BASE_CUDA_T(O)> o_ptr((BASE_CUDA_T(O)*)mat.data, mat.cols, mat.rows, mat.step);
        fk_output.push_back(o_ptr);
    }
    return internal::split_builder_t<O, fk::Ptr2D<BASE_CUDA_T(O)>, fk::WriteDeviceFunction<fk::SplitWrite<fk::_2D, CUDA_T(O)>>>::build(fk_output);
}

template <int O>
inline constexpr auto split(const cv::cuda::GpuMat& output, const cv::Size& planeDims) {
    assert(output.cols % (planeDims.width * planeDims.height) == 0 && output.cols / (planeDims.width * planeDims.height) == CV_MAT_CN(O) &&
    "Each row of the GpuMap should contain as many planes as width / (planeDims.width * planeDims.height)");

    const fk::Tensor<BASE_CUDA_T(O)> t_output((BASE_CUDA_T(O)*)output.data, planeDims.width, planeDims.height, output.rows, CV_MAT_CN(O));

    return fk::WriteDeviceFunction<fk::TensorSplitWrite<CUDA_T(O)>> {t_output};
}

template <int O>
inline constexpr auto split(const fk::RawPtr<fk::_3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::WriteDeviceFunction<fk::TensorSplitWrite<CUDA_T(O)>> {output};
}

template <int T, int INTER_F>
inline const auto resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const fk::RawPtr<fk::_2D, CUDA_T(T)> fk_input = 
    {(CUDA_T(T)*)input.data, {(uint)input.cols, (uint)input.rows, (uint)input.step}};

    const fk::Size dSize{dsize.width, dsize.height};
    return fk::resize<CUDA_T(T), (fk::InterpolationType)INTER_F>(fk_input, dSize, (const double)fx, (const double)fy);
}

template <int T, int INTER_F, int NPtr, AspectRatio AR = IGNORE_AR>
inline const auto resize(const std::array<cv::cuda::GpuMat, NPtr>& input, const cv::Size& dsize, const int& usedPlanes, const CUDA_T(T)& backgroundValue = fk::make_set<CUDA_T(T)>(0)) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const std::array<fk::Ptr2D<CUDA_T(T)>, NPtr> fk_input{ [](const std::array<cv::cuda::GpuMat, NPtr>& input) {
            std::array<fk::Ptr2D<CUDA_T(T)>, NPtr> temp;
            std::transform(input.begin(), input.end(), temp.begin(), [](const cv::cuda::GpuMat& i) {
                    return fk::Ptr2D((CUDA_T(T)*)i.data, i.cols, i.rows, i.step);
                });
            return temp;
        }(input)};

    const fk::Size dSize{dsize.width, dsize.height};

    return fk::resize<CUDA_T(T), (fk::InterpolationType)INTER_F, NPtr, (fk::AspectRatio)AR>(fk_input, dSize, usedPlanes, backgroundValue);
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output) {
    const fk::Ptr2D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);
    return fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>>{ fk_output };
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output, const cv::Size& plane) {
    const fk::Tensor<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, plane.width, plane.height, output.rows);
    return fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, CUDA_T(O)>>{ fk_output };
}

template <typename T>
inline constexpr auto write(const fk::Tensor<T>& output) {
    return fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, T>>{ output };
}

template <typename Operation, typename... operations>
inline dim3 extractDataDims(const fk::ReadDeviceFunction<Operation>& op, const operations&... ops) {
    return op.activeThreads;
}

template <typename... operations>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const operations&... ops) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    fk::executeOperations(cu_stream, ops...);
}

template <int I, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, const operations&... ops) {

    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    const fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);

    fk::executeOperations(fk_input, cu_stream, ops...);
}

template <int I, int O, typename... operations>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, const operations&... ops) {

    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    
    const fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input.data, input.cols, input.rows, input.step);
    const fk::Ptr2D<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, output.cols, output.rows, output.step);

    const dim3 block = fk_input.getBlockSize();
    const dim3 grid(ceil(fk_input.dims().width / (float)block.x), ceil(fk_input.dims().height / (float)block.y));
    const dim3 gridActiveThreads(fk_input.dims().width, fk_input.dims().height);
    
    const fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> firstOp { fk_input, gridActiveThreads };
    const fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>> opFinal { fk_output };
    
    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(firstOp, ops..., opFinal);
    
    gpuErrchk(cudaGetLastError());
}

// Batch reads
template <int I, int Batch, typename... operations>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, const int& activeBatch, const cv::cuda::Stream& stream, const operations&... ops) {

    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    const fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input[0].data, input[0].cols, input[0].rows, input[0].step);

    const dim3 block = fk_input.getBlockSize();
    dim3 grid;
    grid.x = (unsigned int)ceil(fk_input.dims().width / (float)block.x);
    grid.y = (unsigned int)ceil(fk_input.dims().height / (float)block.y);
    grid.z = activeBatch;
    const dim3 gridActiveThreads(fk_input.dims().width, fk_input.dims().height, activeBatch);
    fk::ReadDeviceFunction<fk::BatchRead<fk::PerThreadRead<fk::_2D, CUDA_T(I)>, Batch>> firstOp;
    firstOp.params[0] = fk_input;
    for (int plane=1; plane<activeBatch; plane++) {
        const fk::Ptr2D<CUDA_T(I)> fk_input_t((CUDA_T(I)*)input[plane].data, input[plane].cols, input[plane].rows, input[plane].step); 
        firstOp.params[plane] = fk_input_t;
    }
    firstOp.activeThreads = gridActiveThreads;

    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(firstOp, ops...);
    gpuErrchk(cudaGetLastError());
}

template <int I, int O, int Batch, typename... operations>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, const int& activeBatch, const cv::cuda::GpuMat& output, const cv::cuda::Stream& stream, const operations&... ops) {

    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);

    const fk::Tensor<CUDA_T(O)> fk_output((CUDA_T(O)*)output.data, input[0].cols, input[0].rows, Batch);

    const dim3 block = fk_output.getBlockSize();
    const dim3 grid(ceil(input[0].cols / (float)block.x), ceil(input[0].rows / (float)block.y), activeBatch);
    const dim3 gridActiveThreads(input[0].cols, input[0].rows, activeBatch);
    
    fk::ReadDeviceFunction<fk::BatchRead<fk::PerThreadRead<fk::_2D, CUDA_T(I)>, Batch>> firstOp;
    for (int plane=0; plane<activeBatch; plane++) {
        const fk::Ptr2D<CUDA_T(I)> fk_input((CUDA_T(I)*)input[plane].data, input[plane].cols, input[plane].rows, input[plane].step); 
        firstOp.params[plane] = fk_input;
    }
    firstOp.activeThreads = gridActiveThreads;
    const fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, CUDA_T(O)>> opFinal { fk_output };
    
    fk::cuda_transform<<<grid, block, 0, cu_stream>>>(firstOp, ops..., opFinal);
    
    gpuErrchk(cudaGetLastError());
}

/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

template <int I, int O, int COLOR_PLANES, int BATCH>
class CircularTensor : public fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH> {
public:
    inline constexpr CircularTensor() {};

    inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH>(width_, height_, deviceID_) {};

    inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH>::Alloc(width_, height_, deviceID_);
    }

    template <typename... DeviceFunctionTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const cv::cuda::GpuMat& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
        const fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> readDeviceFunction{
            { (CUDA_T(I)*)input.data, { static_cast<uint>(input.cols), static_cast<uint>(input.rows), static_cast<uint>(input.step) } },
            { static_cast<uint>(input.cols), static_cast<uint>(input.rows), 1 }
        };
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH>::update(cv::cuda::StreamAccessor::getStream(stream), readDeviceFunction, deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctionInstances) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH>::update(cv::cuda::StreamAccessor::getStream(stream), deviceFunctionInstances...);
    }

    inline constexpr CUDA_T(O)* data() {
        return this->ptr_a.data;
    }
};
} // namespace cvGS
