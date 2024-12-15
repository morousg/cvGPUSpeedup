/*
   Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <array>

using namespace fk;

int launch() {
    constexpr int BATCH = 20;
    std::array<RawPtr<_2D, float>, BATCH> inputs{};
    std::array<ResizeReadParams<InterpolationType::INTER_LINEAR>, BATCH> resParams{};
    std::array<ApplyROIParams<float>, BATCH> roiParams{};

    const auto batchRead =
        BatchReadBack<BATCH>::build<ApplyROI<ROI::OFFSET_THREADS>,
                                    ResizeRead<InterpolationType::INTER_LINEAR>,
                                    PerThreadRead<_2D, float>>(roiParams, resParams, inputs);

    return 0;
}
