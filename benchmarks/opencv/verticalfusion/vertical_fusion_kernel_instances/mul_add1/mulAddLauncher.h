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

#ifndef MUL_ADD_LAUNCHER_H
#define MUL_ADD_LAUNCHER_H

#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd2-1002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd1102-2002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd2102-3002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd3102-4002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd4102-5002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd5102-6002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd6102-7002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd7102-8002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd8102-9002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd9102-10002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd10102-11002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd11102-12002.h>
#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add1/mulAdd12102-13002.h>

template <int NumOps>
void launchMulAdd(const std::array<cv::cuda::GpuMat, REAL_BATCH>& crops,
    const cv::cuda::Stream& cv_stream,
    const float& alpha,
    const cv::cuda::GpuMat& d_tensor_output,
    const cv::Size& cropSize,
    const MulAddFuncType& dFunc) {
    if constexpr (NumOps == 2) {
        launchMulAdd2(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc);
    }
#define LAUNCH_MUL_ADD(n) else if constexpr (NumOps == n) { launchMulAdd##n(crops, cv_stream, alpha, d_tensor_output, cropSize, dFunc); }
    LAUNCH_MUL_ADD(102)
    LAUNCH_MUL_ADD(202)
    LAUNCH_MUL_ADD(302)
    LAUNCH_MUL_ADD(402)
    LAUNCH_MUL_ADD(502)
    LAUNCH_MUL_ADD(602)
    LAUNCH_MUL_ADD(702)
    LAUNCH_MUL_ADD(802)
    LAUNCH_MUL_ADD(902)
    LAUNCH_MUL_ADD(1002)
    LAUNCH_MUL_ADD(1102)
    LAUNCH_MUL_ADD(1202)
    LAUNCH_MUL_ADD(1302)
    LAUNCH_MUL_ADD(1402)
    LAUNCH_MUL_ADD(1502)
    LAUNCH_MUL_ADD(1602)
    LAUNCH_MUL_ADD(1702)
    LAUNCH_MUL_ADD(1802)
    LAUNCH_MUL_ADD(1902)
    LAUNCH_MUL_ADD(2002)
    LAUNCH_MUL_ADD(2102)
    LAUNCH_MUL_ADD(2202)
    LAUNCH_MUL_ADD(2302)
    LAUNCH_MUL_ADD(2402)
    LAUNCH_MUL_ADD(2502)
    LAUNCH_MUL_ADD(2602)
    LAUNCH_MUL_ADD(2702)
    LAUNCH_MUL_ADD(2802)
    LAUNCH_MUL_ADD(2902)
    LAUNCH_MUL_ADD(3002)
    LAUNCH_MUL_ADD(3102)
    LAUNCH_MUL_ADD(3202)
    LAUNCH_MUL_ADD(3302)
    LAUNCH_MUL_ADD(3402)
    LAUNCH_MUL_ADD(3502)
    LAUNCH_MUL_ADD(3602)
    LAUNCH_MUL_ADD(3702)
    LAUNCH_MUL_ADD(3802)
    LAUNCH_MUL_ADD(3902)
    LAUNCH_MUL_ADD(4002)
    LAUNCH_MUL_ADD(4102)
    LAUNCH_MUL_ADD(4202)
    LAUNCH_MUL_ADD(4302)
    LAUNCH_MUL_ADD(4402)
    LAUNCH_MUL_ADD(4502)
    LAUNCH_MUL_ADD(4602)
    LAUNCH_MUL_ADD(4702)
    LAUNCH_MUL_ADD(4802)
    LAUNCH_MUL_ADD(4902)
    LAUNCH_MUL_ADD(5002)
    LAUNCH_MUL_ADD(5102)
    LAUNCH_MUL_ADD(5202)
    LAUNCH_MUL_ADD(5302)
    LAUNCH_MUL_ADD(5402)
    LAUNCH_MUL_ADD(5502)
    LAUNCH_MUL_ADD(5602)
    LAUNCH_MUL_ADD(5702)
    LAUNCH_MUL_ADD(5802)
    LAUNCH_MUL_ADD(5902)
    LAUNCH_MUL_ADD(6002)
    LAUNCH_MUL_ADD(6102)
    LAUNCH_MUL_ADD(6202)
    LAUNCH_MUL_ADD(6302)
    LAUNCH_MUL_ADD(6402)
    LAUNCH_MUL_ADD(6502)
    LAUNCH_MUL_ADD(6602)
    LAUNCH_MUL_ADD(6702)
    LAUNCH_MUL_ADD(6802)
    LAUNCH_MUL_ADD(6902)
    LAUNCH_MUL_ADD(7002)
    LAUNCH_MUL_ADD(7102)
    LAUNCH_MUL_ADD(7202)
    LAUNCH_MUL_ADD(7302)
    LAUNCH_MUL_ADD(7402)
    LAUNCH_MUL_ADD(7502)
    LAUNCH_MUL_ADD(7602)
    LAUNCH_MUL_ADD(7702)
    LAUNCH_MUL_ADD(7802)
    LAUNCH_MUL_ADD(7902)
    LAUNCH_MUL_ADD(8002)
    LAUNCH_MUL_ADD(8102)
    LAUNCH_MUL_ADD(8202)
    LAUNCH_MUL_ADD(8302)
    LAUNCH_MUL_ADD(8402)
    LAUNCH_MUL_ADD(8502)
    LAUNCH_MUL_ADD(8602)
    LAUNCH_MUL_ADD(8702)
    LAUNCH_MUL_ADD(8802)
    LAUNCH_MUL_ADD(8902)
    LAUNCH_MUL_ADD(9002)
    LAUNCH_MUL_ADD(9102)
    LAUNCH_MUL_ADD(9202)
    LAUNCH_MUL_ADD(9302)
    LAUNCH_MUL_ADD(9402)
    LAUNCH_MUL_ADD(9502)
    LAUNCH_MUL_ADD(9602)
    LAUNCH_MUL_ADD(9702)
    LAUNCH_MUL_ADD(9802)
    LAUNCH_MUL_ADD(9902)
    LAUNCH_MUL_ADD(10002)
    LAUNCH_MUL_ADD(10102)
    LAUNCH_MUL_ADD(10202)
    LAUNCH_MUL_ADD(10302)
    LAUNCH_MUL_ADD(10402)
    LAUNCH_MUL_ADD(10502)
    LAUNCH_MUL_ADD(10602)
    LAUNCH_MUL_ADD(10702)
    LAUNCH_MUL_ADD(10802)
    LAUNCH_MUL_ADD(10902)
    LAUNCH_MUL_ADD(11002)
    LAUNCH_MUL_ADD(11102)
    LAUNCH_MUL_ADD(11202)
    LAUNCH_MUL_ADD(11302)
    LAUNCH_MUL_ADD(11402)
    LAUNCH_MUL_ADD(11502)
    LAUNCH_MUL_ADD(11602)
    LAUNCH_MUL_ADD(11702)
    LAUNCH_MUL_ADD(11802)
    LAUNCH_MUL_ADD(11902)
    LAUNCH_MUL_ADD(12002)
    LAUNCH_MUL_ADD(12102)
    LAUNCH_MUL_ADD(12202)
    LAUNCH_MUL_ADD(12302)
    LAUNCH_MUL_ADD(12402)
    LAUNCH_MUL_ADD(12502)
    LAUNCH_MUL_ADD(12602)
    LAUNCH_MUL_ADD(12702)
    LAUNCH_MUL_ADD(12802)
    LAUNCH_MUL_ADD(12902)
    LAUNCH_MUL_ADD(13002)
#undef LAUNCH_MUL_ADD
}

#endif // MUL_ADD_LAUNCHER_H
