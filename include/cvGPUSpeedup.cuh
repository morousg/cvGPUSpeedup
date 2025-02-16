/* Copyright 2025 Grup Mediapro, S.L.U.
   Copyright 2023-2025 Oscar Amoros Huguet

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
#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/core/data/circular_tensor.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/algorithms/image_processing/color_conversion.cuh>
#include <fused_kernel/algorithms/image_processing/crop.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvGS {

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2, PRESERVE_AR_LEFT = 3 };

template <typename T>
inline constexpr fk::Ptr2D<T> gpuMat2Ptr2D(const cv::cuda::GpuMat& source) {
    const fk::Ptr2D<T> temp(reinterpret_cast<T*>(source.data), source.cols, source.rows, (uint)source.step);
    return temp;
}

template <typename T>
inline constexpr fk::RawPtr<fk::_2D, T> gpuMat2RawPtr2D(const cv::cuda::GpuMat& source) {
    const fk::RawPtr<fk::_2D, T> temp{ (T*)source.data, {static_cast<uint>(source.cols), static_cast<uint>(source.rows), static_cast<uint>(source.step)} };
    return temp;
}

template <typename T, int Batch>
inline constexpr std::array<fk::Ptr2D<T>, Batch> gpuMat2Ptr2D_arr(const std::array<cv::cuda::GpuMat, Batch>& source) {
    std::array<fk::Ptr2D<T>, Batch> temp;
    std::transform(source.begin(), source.end(), temp.begin(),
                        [](const cv::cuda::GpuMat& i) { return gpuMat2Ptr2D<T>(i); });
    return temp;
}

template <typename T, int Batch>
inline constexpr std::array<fk::RawPtr<fk::_2D, T>, Batch> gpuMat2RawPtr2D_arr(const std::array<cv::cuda::GpuMat, Batch>& source) {
    std::array<fk::RawPtr<fk::_2D, T>, Batch> temp;
    std::transform(source.begin(), source.end(), temp.begin(),
        [](const cv::cuda::GpuMat& i) { return gpuMat2RawPtr2D<T>(i); });
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

    using FirstOp = fk::SaturateCast<CUDA_T(I), CUDA_T(O)>;
    using SecondOp = fk::Mul<CUDA_T(O)>;
    return FirstOp::build().then(SecondOp::build(fk::make_set<CUDA_T(O)>(alpha)));
}

template <int I, int O>
inline constexpr auto convertTo(float alpha, float beta) {
    using InputBase = typename fk::VectorTraits<CUDA_T(I)>::base;
    using OutputBase = typename fk::VectorTraits<CUDA_T(O)>::base;

    using FirstOp = fk::SaturateCast<CUDA_T(I), CUDA_T(O)>;
    using SecondOp = fk::Mul<CUDA_T(O)>;
    using ThirdOp = fk::Add<CUDA_T(O)>;

    return fk::FusedOperation<FirstOp, SecondOp, ThirdOp>::build({ { {fk::make_set<CUDA_T(O)>(alpha), { fk::make_set<CUDA_T(O)>(beta) }} } });
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
    return fk::Binary<fk::Add<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <cv::ColorConversionCodes CODE, int I, int O = I>
inline constexpr auto cvtColor() {
    static_assert((CV_MAT_DEPTH(I) == CV_8U || CV_MAT_DEPTH(I) == CV_16U || CV_MAT_DEPTH(I) == CV_32F) &&
                  (CV_MAT_DEPTH(O) == CV_8U || CV_MAT_DEPTH(O) == CV_16U || CV_MAT_DEPTH(O) == CV_32F),
                  "Wrong CV_TYPE_DEPTH, it has to be CV_8U, or CV_16U or CV_32F");
    static_assert(isSupportedColorConversion<CODE>, "Color conversion type not supported yet.");
    using InputType = CUDA_T(I);
    using OutputType = CUDA_T(O);

    return fk::Unary<fk::ColorConversion<(fk::ColorConversionCodes)CODE, InputType, OutputType>>{};
}

template <int O>
inline constexpr auto split(const std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
    for (auto& mat : output) {
        fk_output.push_back(gpuMat2Ptr2D<BASE_CUDA_T(O)>(mat));
    }
    return fk::SplitWrite<fk::_2D, CUDA_T(O)>::build(fk_output);
}

template <int O, int N>
inline constexpr auto split(const std::array<std::vector<cv::cuda::GpuMat>, N>& output) {
    std::array<std::vector<fk::Ptr2D<BASE_CUDA_T(O)>>, N> fkOutput{};
    for (int i = 0; i < N; ++i) {
        std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
        for (auto& mat : output[i]) {
            fk_output.push_back(gpuMat2Ptr2D<BASE_CUDA_T(O)>(mat));
        }
        fkOutput[i] = fk_output;
    }
    return fk::SplitWrite<fk::_2D, CUDA_T(O)>::build(fkOutput);
}

template <int O>
inline constexpr auto split(const cv::cuda::GpuMat& output, const cv::Size& planeDims) {
    assert(output.cols % (planeDims.width * planeDims.height) == 0 && output.cols / (planeDims.width * planeDims.height) == CV_MAT_CN(O) &&
    "Each row of the GpuMap should contain as many planes as width / (planeDims.width * planeDims.height)");

    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {
        gpuMat2Tensor<BASE_CUDA_T(O)>(output, planeDims, CV_MAT_CN(O)).ptr()};
}

template <int O>
inline constexpr auto split(const fk::RawPtr<fk::_3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {output};
}

template <int O>
inline constexpr auto splitT(const fk::RawPtr<fk::T3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorTSplit<CUDA_T(O)>> {output};
}

template <int INTER_F>
inline const auto resize(const cv::Size& dsize) {
    return fk::ResizeRead<static_cast<fk::InterpolationType>(INTER_F)>::build(fk::Size(dsize.width, dsize.height));
}

template <int T, int INTER_F>
inline const auto resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const fk::RawPtr<fk::_2D, CUDA_T(T)> fk_input = gpuMat2Ptr2D<CUDA_T(T)>(input);
    const fk::Size dSize{ dsize.width, dsize.height };
    return fk::ResizeRead<(fk::InterpolationType)INTER_F>::build(fk_input, dSize, fx, fy);
}

template <int T, int INTER_F, int NPtr, AspectRatio AR_ = IGNORE_AR>
inline const auto resize(const std::array<cv::cuda::GpuMat, NPtr>& input,
                         const cv::Size& dsize, const int& usedPlanes,
                         const cv::Scalar& backgroundValue_ = cvScalar_set<CV_MAKETYPE(CV_32F, CV_MAT_CN(T))>(0)) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    constexpr fk::AspectRatio AR = static_cast<fk::AspectRatio>(AR_);
    constexpr fk::InterpolationType IType = static_cast<fk::InterpolationType>(INTER_F);

    constexpr int defaultType = CV_MAKETYPE(CV_32F, CV_MAT_CN(T));
    const std::array<fk::RawPtr<fk::_2D, CUDA_T(T)>, NPtr> fk_input{ gpuMat2RawPtr2D_arr<CUDA_T(T), NPtr>(input) };

    using PixelReadOp = fk::PerThreadRead<fk::_2D, CUDA_T(T)>;
    using O = CUDA_T(defaultType);
    const O backgroundValue = cvScalar2CUDAV<defaultType>::get(backgroundValue_);
    
    const auto readOP = PixelReadOp::build_batch(fk_input);
    const auto sizeArr = fk::make_set_std_array<NPtr>(fk::Size(dsize.width, dsize.height));
    const auto backgroundArr = fk::make_set_std_array<NPtr>(backgroundValue);
    using Resize = fk::ResizeRead<IType, AR, fk::Read<PixelReadOp>>;
    if constexpr (AR != fk::IGNORE_AR) {
        const auto resizeDFs = Resize::build_batch(readOP, sizeArr, backgroundArr);
        return fk::BatchRead<NPtr, fk::CONDITIONAL_WITH_DEFAULT>::build(resizeDFs, usedPlanes, backgroundValue);
    } else {
        const auto resizeDFs = Resize::build_batch(readOP, sizeArr);
        return fk::BatchRead<NPtr, fk::CONDITIONAL_WITH_DEFAULT>::build(resizeDFs, usedPlanes, backgroundValue);
    }
}

inline constexpr auto crop(const cv::Rect2d& rect) {
    return fk::Crop<>::build(fk::Rect(static_cast<uint>(rect.x), static_cast<uint>(rect.y), static_cast<int>(rect.width), static_cast<int>(rect.height)));
}

template <int BATCH>
inline constexpr auto crop(const std::array<cv::Rect2d, BATCH>& rects) {
    std::array<fk::Rect, BATCH> fk_rects{};

    for (int i = 0; i < BATCH; ++i) {
        const auto tmp = rects[i];
        fk_rects[i] = fk::Rect(static_cast<uint>(tmp.x), static_cast<uint>(tmp.y), static_cast<int>(tmp.width), static_cast<int>(tmp.height));
    }
    return fk::Crop<>::build(fk_rects);
}

template <typename BackIOp>
inline constexpr auto crop(const BackIOp& backIOp, const cv::Rect2d& rect) {
    return fk::Crop<BackIOp>::build(backIOp, fk::Rect(static_cast<uint>(rect.x), static_cast<uint>(rect.y), static_cast<int>(rect.width), static_cast<int>(rect.height)));
}

template <typename BackIOp, int BATCH>
inline constexpr auto crop(const BackIOp& backIOp, const std::array<cv::Rect2d, BATCH>& rects) {
    return backIOp.then(crop(rects));
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output) {
    return fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>>{ gpuMat2Ptr2D<CUDA_T(O)>(output).ptr() };
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output, const cv::Size& plane) {
    return fk::Write<fk::PerThreadWrite<fk::_3D, CUDA_T(O)>>{ gpuMat2Tensor<CUDA_T(O)>(output, plane, 1).ptr() };
}

template <typename T>
inline constexpr auto write(const fk::Tensor<T>& output) {
    return fk::WriteInstantiableOperation<fk::PerThreadWrite<fk::_3D, T>>{ output };
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    fk::executeOperations<ENABLE_THREAD_FUSION>(cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D<InputType>(input), cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                        cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D<InputType>(input),
                                                gpuMat2Ptr2D<OutputType>(output), cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                        cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, output, stream, instantiableOperations...);
}

// Batch reads
template <bool ENABLE_THREAD_FUSION, size_t Batch,  typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    // On Linux (gcc 11.4) it is necessary to pass the InputType and Batch as a template parameter
    // On Windows (VS2022 Community) it is not needed, it is deduced from the parameters being passed
    fk::executeOperations<ENABLE_THREAD_FUSION, InputType, Batch>(gpuMat2Ptr2D_arr<InputType, Batch>(input),
                                                                  activeBatch, cvScalar2CUDAV_t<InputType>::get(defaultValue),
                                                                  cu_stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    // On Linux (gcc 11.4) it is necessary to pass the InputType and Batch as a template parameter
    // On Windows (VS2022 Community) it is not needed, it is deduced from the parameters being passed
    fk::executeOperations<ENABLE_THREAD_FUSION, InputType, Batch>(gpuMat2Ptr2D_arr<InputType, Batch>(input),
                                                                  cu_stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, activeBatch, defaultValue, stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType>(input),
                                                activeBatch, cvScalar2CUDAV_t<InputType>::get(defaultValue),
                                                gpuMat2Tensor<OutputType>(output, outputPlane, 1),
                                                cu_stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType>(input),
                                                gpuMat2Tensor<OutputType>(output, outputPlane, 1),
                                                cu_stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, 
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, activeBatch, defaultValue, output, outputPlane, stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, output, outputPlane, stream, instantiableOperations...);
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

    template <typename... IOpTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const cv::cuda::GpuMat& input, const IOpTypes&... instantiableOperationInstances) {
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> readInstantiableOperation{
            {{(CUDA_T(I)*)input.data, { static_cast<uint>(input.cols), static_cast<uint>(input.rows), static_cast<uint>(input.step) }}}};
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), readInstantiableOperation, instantiableOperationInstances...);
    }

    template <typename... IOpTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperationInstances) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), instantiableOperationInstances...);
    }

    inline constexpr CUDA_T(O)* data() {
        return this->ptr_a.data;
    }
};
} // namespace cvGS
