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

#ifndef LAUNCH_MUL_ADD_MACRO_CUH
#define LAUNCH_MUL_ADD_MACRO_CUH

#include <benchmarks/opencv/verticalfusion/vertical_fusion_static_loop.cuh>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add/mulAddOpType.cuh>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul/realBatch.h>

#define LAUNCH(NumOps) \
void launchMulAdd##NumOps(const std::array<cv::cuda::GpuMat, REAL_BATCH>& crops, \
    const cv::cuda::Stream& cv_stream, \
    const float& alpha, \
    const cv::cuda::GpuMat& d_tensor_output, \
    const cv::Size& cropSize, \
    const MulAddFuncType& dFunc) { \
    VerticalFusion<CV_8UC1, CV_32FC1, 2, NumOps, MulAddFuncType>::execute(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc); \
}

#endif // LAUNCH_MUL_ADD_MACRO_CUH
