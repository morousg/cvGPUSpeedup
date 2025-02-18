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

#include <benchmarks/opencv/verticalfusion/vertical_fusion_kernel_instances/mul_add/launchMulAddHeaderMacro.cuh>

LAUNCH_MUL_HEADER(2)
LAUNCH_MUL_HEADER(102)
LAUNCH_MUL_HEADER(202)
LAUNCH_MUL_HEADER(302)
LAUNCH_MUL_HEADER(402)
LAUNCH_MUL_HEADER(502)
LAUNCH_MUL_HEADER(602)
LAUNCH_MUL_HEADER(702)
LAUNCH_MUL_HEADER(802)
LAUNCH_MUL_HEADER(902)
LAUNCH_MUL_HEADER(1002)

#undef LAUNCH_MUL_HEADER
