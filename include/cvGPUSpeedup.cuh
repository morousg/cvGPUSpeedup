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

#include <cvGPUSpeedupHelpers.cuh>
#include <fused_kernel/fused_kernel_launchers.cuh>
#include <fused_kernel/device_function_builders.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>

namespace cvGS {

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1 };

template <typename T>
inline constexpr fk::Ptr2D<T> gpuMat2Ptr2D(const cv::cuda::GpuMat& source) {
    const fk::Ptr2D<T> temp((T*)source.data, source.cols, source.rows, (uint)source.step);
    return temp;
}

template <typename T, int Batch>
inline constexpr std::array<fk::Ptr2D<T>, Batch> gpuMat2Ptr2D_arr(const std::array<cv::cuda::GpuMat, Batch>& source) {
    std::array<fk::Ptr2D<T>, Batch> temp;
    std::transform(source.begin(), source.end(), temp.begin(),
                        [](const cv::cuda::GpuMat& i) { return gpuMat2Ptr2D<T>(i); });
    return temp;
}

template <typename T>
inline constexpr fk::Tensor<T> gpuMat2Tensor(const cv::cuda::GpuMat& source, const cv::Size& planeDims, const int& colorPlanes) {
    const fk::Tensor<T> t_output((T*)source.data, planeDims.width, planeDims.height, source.rows, colorPlanes);
    return t_output;
}

template <int I, int O>
inline constexpr auto convertTo() {
    return fk::Unary<fk::SaturateCast<CUDA_T(I), CUDA_T(O)>>{};
}

template <int I, int O>
inline constexpr auto convertTo(float alpha) {
    using InputBase = typename fk::VectorTraits<CUDA_T(I)>::base;
    using OutputBase = typename fk::VectorTraits<CUDA_T(O)>::base;

    using FirstOp = fk::Unary<fk::SaturateCast<CUDA_T(I), CUDA_T(O)>>;
    using SecondOp = fk::Binary<fk::Mul<CUDA_T(O)>>;
    return fk::Binary<fk::ComposedOperation<FirstOp, SecondOp>>{{{}, { fk::make_set<CUDA_T(O)>(alpha) }}};
}

template <int I, int O>
inline constexpr auto convertTo(float alpha, float beta) {
    using InputBase = typename fk::VectorTraits<CUDA_T(I)>::base;
    using OutputBase = typename fk::VectorTraits<CUDA_T(O)>::base;

    using FirstOp = fk::Unary<fk::SaturateCast<CUDA_T(I), CUDA_T(O)>>;
    using SecondOp = fk::Binary<fk::Mul<CUDA_T(O)>>;
    using ThirdOp = fk::Binary<fk::Sum<CUDA_T(O)>>;
    return fk::Binary<fk::ComposedOperation<FirstOp, SecondOp, ThirdOp>>{{{}, { fk::make_set<CUDA_T(O)>(alpha) }, { fk::make_set<CUDA_T(O)>(beta) }}};
}

template <int I>
inline constexpr auto multiply(const cv::Scalar& src2) {
    return fk::Binary<fk::Mul<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto subtract(const cv::Scalar& src2) {
    return fk::Binary<fk::Sub<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto divide(const cv::Scalar& src2) {
    return fk::Binary<fk::Div<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto add(const cv::Scalar& src2) {
    return fk::Binary<fk::Sum<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I, int O, cv::ColorConversionCodes CODE>
inline constexpr auto cvtColor() {
    static_assert((CV_MAT_DEPTH(I) == CV_8U || CV_MAT_DEPTH(I) == CV_16U || CV_MAT_DEPTH(I) == CV_32F) &&
                  (CV_MAT_DEPTH(O) == CV_8U || CV_MAT_DEPTH(O) == CV_16U || CV_MAT_DEPTH(O) == CV_32F),
                  "Wrong CV_TYPE_DEPTH, it has to be CV_8U, or CV_16U or CV_32F");
    static_assert(isSupportedColorConversion<CODE>, "Color conversion type not supported yet.");
    using InputType = CUDA_T(I);
    using OutputType = CUDA_T(O);
    using BaseIT = BASE_CUDA_T(I);
    if constexpr (CODE == cv::COLOR_BGR2BGRA || CODE == cv::COLOR_RGB2RGBA) {
        static_assert(std::is_same_v<fk::VBase<InputType>, fk::VBase<OutputType>>,"cvtColor does not support different input and otput base types");
        using DeviceFunctionType = fk::Binary<fk::AddLast<InputType, typename fk::VectorType<BaseIT, fk::cn<InputType> +1>::type>>;
        if constexpr (CV_MAT_DEPTH(I) == CV_8U) {
            return DeviceFunctionType{ fk::maxValue<uchar> };
        } else if constexpr (CV_MAT_DEPTH(I) == CV_16U) {
            return DeviceFunctionType{ fk::maxValue<ushort> };
        } else if constexpr (CV_MAT_DEPTH(I) == CV_32F) {
            return DeviceFunctionType{ 1.f };
        }
    } else if constexpr (CODE == cv::COLOR_BGRA2BGR || CODE == cv::COLOR_RGBA2RGB) {
        static_assert(std::is_same_v<fk::VBase<InputType>, fk::VBase<OutputType>>, "cvtColor does not support different input and otput base types");
        return fk::Unary<fk::Discard<InputType, typename fk::VectorType<BaseIT, 3>::type>>{};
    } else if constexpr (CODE == cv::COLOR_BGR2RGBA || CODE == cv::COLOR_RGB2BGRA) {
        static_assert(std::is_same_v<fk::VBase<InputType>, fk::VBase<OutputType>>, "cvtColor does not support different input and otput base types");
        using FirstDeviceFunctionType = fk::Unary<fk::VectorReorder<InputType, 2, 1, 0>>;
        using SecondDeviceFunctionType = 
            fk::Binary<fk::AddLast<InputType, typename fk::VectorType<BaseIT, fk::cn<InputType> +1>::type>>;
        using DeviceFunctionType = 
            fk::Binary<fk::ComposedOperation<FirstDeviceFunctionType, SecondDeviceFunctionType>>;
        if constexpr (CV_MAT_DEPTH(I) == CV_8U) {
            return DeviceFunctionType{ {{}, {fk::maxValue<uchar>}} };
        } else if constexpr (CV_MAT_DEPTH(I) == CV_16U) {
            return DeviceFunctionType{ {{}, {fk::maxValue<ushort>}} };
        } else if constexpr (CV_MAT_DEPTH(I) == CV_32F) {
            return DeviceFunctionType{ {{}, {1.f}} };
        }
    } else if constexpr (CODE == cv::COLOR_BGRA2RGB || CODE == cv::COLOR_RGBA2BGR) {
        static_assert(std::is_same_v<fk::VBase<InputType>, fk::VBase<OutputType>>, "cvtColor does not support different input and otput base types");
        using FirstOperation = fk::VectorReorder<InputType, 2, 1, 0, 3>;
        using SecondOperation = fk::Discard<InputType, typename fk::VectorType<BaseIT, 3>::type>;
        return fk::Unary<fk::OperationSequence<FirstOperation, SecondOperation>> {};
    } else if constexpr (CODE == cv::COLOR_BGR2RGB || CODE == cv::COLOR_RGB2BGR) {
        static_assert(std::is_same_v<InputType, OutputType>, "cvtColor does not support different input and otput types");
        return fk::Unary<fk::VectorReorder<InputType, 2, 1, 0>> {};
    } else if constexpr (CODE == cv::COLOR_BGRA2RGBA || CODE == cv::COLOR_RGBA2BGRA) {
        static_assert(std::is_same_v<InputType, OutputType>, "cvtColor does not support different input and otput types");
        return fk::Unary<fk::VectorReorder<InputType, 2, 1, 0, 3>> {};
    } else if constexpr (CODE == cv::COLOR_RGB2GRAY || CODE == cv::COLOR_RGBA2GRAY) {
        return fk::Unary<fk::RGB2Gray<InputType, OutputType>> {};
    } else if constexpr (CODE == cv::COLOR_BGR2GRAY) {
        using FOperation = fk::VectorReorder<InputType, 2, 1, 0>;
        using SOperation = fk::RGB2Gray<InputType, OutputType>;
        using MyOperation = fk::OperationSequence<FOperation, SOperation>;
        return fk::Unary<MyOperation> {};
    } else if constexpr (CODE == cv::COLOR_BGRA2GRAY) {
        return fk::Unary<fk::OperationSequence<fk::VectorReorder<InputType, 2, 1, 0, 3>, fk::RGB2Gray<InputType, OutputType>>> {};
    }
}

template <int O>
inline constexpr auto split(const std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
    for (auto& mat : output) {
        fk_output.push_back(gpuMat2Ptr2D<BASE_CUDA_T(O)>(mat));
    }
    return internal::split_builder_t<O, fk::Ptr2D<BASE_CUDA_T(O)>, fk::Write<fk::SplitWrite<fk::_2D, CUDA_T(O)>>>::build(fk_output);
}

template <int O>
inline constexpr auto split(const cv::cuda::GpuMat& output, const cv::Size& planeDims) {
    assert(output.cols % (planeDims.width * planeDims.height) == 0 && output.cols / (planeDims.width * planeDims.height) == CV_MAT_CN(O) &&
    "Each row of the GpuMap should contain as many planes as width / (planeDims.width * planeDims.height)");

    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {
        gpuMat2Tensor<BASE_CUDA_T(O)>(output, planeDims, CV_MAT_CN(O))};
}

template <int O>
inline constexpr auto split(const fk::RawPtr<fk::_3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {output};
}

template <int O>
inline constexpr auto splitT(const fk::RawPtr<fk::T3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorTSplit<CUDA_T(O)>> {output};
}

template <int T, int INTER_F>
inline const auto resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const fk::RawPtr<fk::_2D, CUDA_T(T)> fk_input = gpuMat2Ptr2D<CUDA_T(T)>(input);
    const fk::Size dSize{ dsize.width, dsize.height };
    return fk::resize<CUDA_T(T), (fk::InterpolationType)INTER_F>(fk_input, dSize, fx, fy);
}

template <int T, int INTER_F, int NPtr, AspectRatio AR = IGNORE_AR>
inline const auto resize(const std::array<cv::cuda::GpuMat, NPtr>& input, const cv::Size& dsize, const int& usedPlanes, const cv::Scalar& backgroundValue = cvScalar_set<CV_MAKETYPE(CV_32F, CV_MAT_CN(T))>(0)) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const std::array<fk::Ptr2D<CUDA_T(T)>, NPtr> fk_input{ gpuMat2Ptr2D_arr<CUDA_T(T), NPtr>(input) };
    const fk::Size dSize{ dsize.width, dsize.height };
    constexpr int defaultType = CV_MAKETYPE(CV_32F, CV_MAT_CN(T));
    return fk::resize<CUDA_T(T), (fk::InterpolationType)INTER_F, NPtr, (fk::AspectRatio)AR>(fk_input, dSize, usedPlanes, cvScalar2CUDAV<defaultType>::get(backgroundValue));
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output) {
    return fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>>{ gpuMat2Ptr2D<CUDA_T(O)>(output) };
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output, const cv::Size& plane) {
    return fk::Write<fk::PerThreadWrite<fk::_3D, CUDA_T(O)>>{ gpuMat2Tensor<CUDA_T(O)>(output, plane, 1) };
}

template <typename T>
inline constexpr auto write(const fk::Tensor<T>& output) {
    return fk::WriteDeviceFunction<fk::PerThreadWrite<fk::_3D, T>>{ output };
}

template <typename... DeviceFunctionTypes>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctions) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    fk::executeOperations(cu_stream, deviceFunctions...);
}

template <typename... DeviceFunctionTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctions) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstDeviceFunctionInputType_t<DeviceFunctionTypes...>;
    fk::executeOperations(gpuMat2Ptr2D<InputType>(input), cu_stream, deviceFunctions...);
}

template <typename... DeviceFunctionTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctions) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstDeviceFunctionInputType_t<DeviceFunctionTypes...>;
    using OutputType = fk::LastDeviceFunctionOutputType_t<DeviceFunctionTypes...>;
    fk::executeOperations(gpuMat2Ptr2D<InputType>(input), gpuMat2Ptr2D<OutputType>(output), cu_stream, deviceFunctions...);
}

// Batch reads
template <size_t Batch, typename... DeviceFunctionTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, const size_t& activeBatch, const cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctions) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstDeviceFunctionInputType_t<DeviceFunctionTypes...>;
    // On Linux (gcc 11.4) it is necessary to pass the InputType and Batch as a template parameter
    // On Windows (VS2022 Community) it is not needed, it is deduced from the parameters being passed
    fk::executeOperations<InputType, Batch>(gpuMat2Ptr2D_arr<InputType, Batch>(input), activeBatch, cu_stream, deviceFunctions...);
}

template <size_t Batch, typename... DeviceFunctionTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, const size_t& activeBatch, const cv::cuda::GpuMat& output, const cv::Size& outputPlane, const cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctions) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstDeviceFunctionInputType_t<DeviceFunctionTypes...>;
    using OutputType = fk::LastDeviceFunctionOutputType_t<DeviceFunctionTypes...>;
    fk::executeOperations(gpuMat2Ptr2D_arr<InputType>(input), activeBatch, gpuMat2Tensor<OutputType>(output, outputPlane, 1), cu_stream, deviceFunctions...);
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

template <int I, int O, int COLOR_PLANES, int BATCH, fk::CircularTensorOrder CT_ORDER, fk::ColorPlanes CP_MODE = fk::ColorPlanes::Standard>
class CircularTensor : public fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE> {
public:
    inline constexpr CircularTensor() {};

    inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>(width_, height_, deviceID_) {};

    inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::Alloc(width_, height_, deviceID_);
    }

    template <typename... DeviceFunctionTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const cv::cuda::GpuMat& input, const DeviceFunctionTypes&... deviceFunctionInstances) {
        const fk::ReadDeviceFunction<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> readDeviceFunction{
            { (CUDA_T(I)*)input.data, { static_cast<uint>(input.cols), static_cast<uint>(input.rows), static_cast<uint>(input.step) } },
            { static_cast<uint>(input.cols), static_cast<uint>(input.rows), 1 }
        };
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), readDeviceFunction, deviceFunctionInstances...);
    }

    template <typename... DeviceFunctionTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const DeviceFunctionTypes&... deviceFunctionInstances) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), deviceFunctionInstances...);
    }

    inline constexpr CUDA_T(O)* data() {
        return this->ptr_a.data;
    }
};
} // namespace cvGS
