/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul2-1002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_static_loop.cuh>

template <int CV_TYPE_I, int CV_TYPE_O, int OPS_PER_ITER, size_t NumOps, typename DeviceFunction>
void launchVerticalFusion(const std::array<cv::cuda::GpuMat, 50>& crops,
    const cv::cuda::Stream& cv_stream,
    const float& alpha,
    const cv::cuda::GpuMat& d_tensor_output,
    const cv::Size& cropSize,
    const DeviceFunction& dFunc) {
    VerticalFusion<CV_8UC1, CV_32FC1, 2, NumOps, DeviceFunction>::execute(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc);
}

#define LAUNCH(NumOps) \
void launchMul##NumOps(const std::array<cv::cuda::GpuMat, 50>& crops, \
    const cv::cuda::Stream& cv_stream, \
    const float& alpha, \
    const cv::cuda::GpuMat& d_tensor_output, \
    const cv::Size& cropSize, \
    const MulFuncType& dFunc) { \
    launchVerticalFusion<CV_8UC1, CV_32FC1, 2, NumOps, MulFuncType>(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc); \
}

LAUNCH(10102)
LAUNCH(10202)
LAUNCH(10302)
LAUNCH(10402)
LAUNCH(10502)
LAUNCH(10602)
LAUNCH(10702)
LAUNCH(10802)
LAUNCH(10902)
LAUNCH(11002)

#undef LAUNCH