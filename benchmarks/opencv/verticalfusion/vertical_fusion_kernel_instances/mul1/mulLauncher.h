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

#ifndef MUL_LAUNCHER_H
#define MUL_LAUNCHER_H

#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul2-1002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul1102-2002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul2102-3002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul3102-4002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul4102-5002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul5102-6002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul6102-7002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul7102-8002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul8102-9002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul9102-10002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul10102-11002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul11102-12002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul1/mul12102-13002.h>

template <int NumOps>
void launchMul(const std::array<cv::cuda::GpuMat, REAL_BATCH>& crops,
               const cv::cuda::Stream& cv_stream,
               const float& alpha,
               const cv::cuda::GpuMat& d_tensor_output,
               const cv::Size& cropSize,
               const MulFuncType& dFunc) {
    // Not using else if constexpr because VS2022 does not support so many if else constexpr by default
    if constexpr (NumOps > 13002) {
        throw std::runtime_error("Unsupported number of operations");
    }
#define LAUNCH_MUL(n) if constexpr (NumOps == n) { launchMul##n(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc); }
    LAUNCH_MUL(2)
    LAUNCH_MUL(102)
    LAUNCH_MUL(202)
    LAUNCH_MUL(302)
    LAUNCH_MUL(402)
    LAUNCH_MUL(502)
    LAUNCH_MUL(602)
    LAUNCH_MUL(702)
    LAUNCH_MUL(802)
    LAUNCH_MUL(902)
    LAUNCH_MUL(1002)
    LAUNCH_MUL(1102)
    LAUNCH_MUL(1202)
    LAUNCH_MUL(1302)
    LAUNCH_MUL(1402)
    LAUNCH_MUL(1502)
    LAUNCH_MUL(1602)
    LAUNCH_MUL(1702)
    LAUNCH_MUL(1802)
    LAUNCH_MUL(1902)
    LAUNCH_MUL(2002)
    LAUNCH_MUL(2102)
    LAUNCH_MUL(2202)
    LAUNCH_MUL(2302)
    LAUNCH_MUL(2402)
    LAUNCH_MUL(2502)
    LAUNCH_MUL(2602)
    LAUNCH_MUL(2702)
    LAUNCH_MUL(2802)
    LAUNCH_MUL(2902)
    LAUNCH_MUL(3002)
    LAUNCH_MUL(3102)
    LAUNCH_MUL(3202)
    LAUNCH_MUL(3302)
    LAUNCH_MUL(3402)
    LAUNCH_MUL(3502)
    LAUNCH_MUL(3602)
    LAUNCH_MUL(3702)
    LAUNCH_MUL(3802)
    LAUNCH_MUL(3902)
    LAUNCH_MUL(4002)
    LAUNCH_MUL(4102)
    LAUNCH_MUL(4202)
    LAUNCH_MUL(4302)
    LAUNCH_MUL(4402)
    LAUNCH_MUL(4502)
    LAUNCH_MUL(4602)
    LAUNCH_MUL(4702)
    LAUNCH_MUL(4802)
    LAUNCH_MUL(4902)
    LAUNCH_MUL(5002)
    LAUNCH_MUL(5102)
    LAUNCH_MUL(5202)
    LAUNCH_MUL(5302)
    LAUNCH_MUL(5402)
    LAUNCH_MUL(5502)
    LAUNCH_MUL(5602)
    LAUNCH_MUL(5702)
    LAUNCH_MUL(5802)
    LAUNCH_MUL(5902)
    LAUNCH_MUL(6002)
    LAUNCH_MUL(6102)
    LAUNCH_MUL(6202)
    LAUNCH_MUL(6302)
    LAUNCH_MUL(6402)
    LAUNCH_MUL(6502)
    LAUNCH_MUL(6602)
    LAUNCH_MUL(6702)
    LAUNCH_MUL(6802)
    LAUNCH_MUL(6902)
    LAUNCH_MUL(7002)
    LAUNCH_MUL(7102)
    LAUNCH_MUL(7202)
    LAUNCH_MUL(7302)
    LAUNCH_MUL(7402)
    LAUNCH_MUL(7502)
    LAUNCH_MUL(7602)
    LAUNCH_MUL(7702)
    LAUNCH_MUL(7802)
    LAUNCH_MUL(7902)
    LAUNCH_MUL(8002)
    LAUNCH_MUL(8102)
    LAUNCH_MUL(8202)
    LAUNCH_MUL(8302)
    LAUNCH_MUL(8402)
    LAUNCH_MUL(8502)
    LAUNCH_MUL(8602)
    LAUNCH_MUL(8702)
    LAUNCH_MUL(8802)
    LAUNCH_MUL(8902)
    LAUNCH_MUL(9002)
    LAUNCH_MUL(9102)
    LAUNCH_MUL(9202)
    LAUNCH_MUL(9302)
    LAUNCH_MUL(9402)
    LAUNCH_MUL(9502)
    LAUNCH_MUL(9602)
    LAUNCH_MUL(9702)
    LAUNCH_MUL(9802)
    LAUNCH_MUL(9902)
    LAUNCH_MUL(10002)
    LAUNCH_MUL(10102)
    LAUNCH_MUL(10202)
    LAUNCH_MUL(10302)
    LAUNCH_MUL(10402)
    LAUNCH_MUL(10502)
    LAUNCH_MUL(10602)
    LAUNCH_MUL(10702)
    LAUNCH_MUL(10802)
    LAUNCH_MUL(10902)
    LAUNCH_MUL(11002)
    LAUNCH_MUL(11102)
    LAUNCH_MUL(11202)
    LAUNCH_MUL(11302)
    LAUNCH_MUL(11402)
    LAUNCH_MUL(11502)
    LAUNCH_MUL(11602)
    LAUNCH_MUL(11702)
    LAUNCH_MUL(11802)
    LAUNCH_MUL(11902)
    LAUNCH_MUL(12002)
    LAUNCH_MUL(12102)
    LAUNCH_MUL(12202)
    LAUNCH_MUL(12302)
    LAUNCH_MUL(12402)
    LAUNCH_MUL(12502)
    LAUNCH_MUL(12602)
    LAUNCH_MUL(12702)
    LAUNCH_MUL(12802)
    LAUNCH_MUL(12902)
    LAUNCH_MUL(13002)
#undef LAUNCH_MUL
}

#endif // MUL_LAUNCHER_H