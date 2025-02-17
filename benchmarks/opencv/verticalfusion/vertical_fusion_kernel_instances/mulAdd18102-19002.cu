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

#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mulAdd18102-19002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_static_loop.cuh>

#define LAUNCH(NumOps) \
void launchMulAdd##NumOps(const std::array<cv::cuda::GpuMat, 50>& crops, \
    const cv::cuda::Stream& cv_stream, \
    const float& alpha, \
    const cv::cuda::GpuMat& d_tensor_output, \
    const cv::Size& cropSize, \
    const MulAddFuncType& dFunc) { \
    VerticalFusion<CV_8UC1, CV_32FC1, 2, NumOps, MulAddFuncType>::execute(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc); \
}

LAUNCH(18102)
LAUNCH(18202)
LAUNCH(18302)
LAUNCH(18402)
LAUNCH(18502)
LAUNCH(18602)
LAUNCH(18702)
LAUNCH(18802)
LAUNCH(18902)
LAUNCH(19002)

#undef LAUNCH