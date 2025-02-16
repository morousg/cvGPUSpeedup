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

#include <opencv2/cudaimgproc.hpp>
#include <cvGPUSpeedup.cuh>

using MulOutType = float;
using MulFuncType = decltype(std::declval<decltype(fk::Mul<MulOutType>::build(std::declval<MulOutType>()))>()
    .then(std::declval<decltype(fk::Mul<MulOutType>::build(std::declval<MulOutType>()))>()));

#define LAUNCH_MUL_HEADER(NumOps) \
void launchMul##NumOps(const std::array<cv::cuda::GpuMat, 50>& crops, \
    const cv::cuda::Stream& cv_stream, \
    const float& alpha, \
    const cv::cuda::GpuMat& d_tensor_output, \
    const cv::Size& cropSize, \
    const MulFuncType& dFunc);

LAUNCH_MUL_HEADER(4102)
LAUNCH_MUL_HEADER(4202)
LAUNCH_MUL_HEADER(4302)
LAUNCH_MUL_HEADER(4402)
LAUNCH_MUL_HEADER(4502)
LAUNCH_MUL_HEADER(4602)
LAUNCH_MUL_HEADER(4702)
LAUNCH_MUL_HEADER(4802)
LAUNCH_MUL_HEADER(4902)
LAUNCH_MUL_HEADER(5002)

#undef LAUNCH_MUL_HEADER