/* Copyright 2023-2025 Oscar Amoros Huguet

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
#include <fused_kernel/core/data/array.cuh>
#include <array>

using namespace fk;

int launch() {
    constexpr int BATCH = 20;
    constexpr RawPtr<_2D, float> data{ nullptr,{16,16,16} };
    constexpr std::array<RawPtr<_2D, float>, BATCH> inputs = make_set_std_array<BATCH>(data);
    constexpr Size oneSize(8,8);
    constexpr std::array<Size, BATCH> resParams = make_set_std_array<BATCH>(oneSize);

    constexpr float defaultValue = 0;
    constexpr std::array<float, BATCH> defaultArray = make_set_std_array<BATCH>(defaultValue);
    constexpr int usedPlanes = 15;

    constexpr auto readDFArray = PerThreadRead<_2D, float>::build_batch<BATCH>(inputs);

    constexpr auto oneResizeread = ResizeRead<INTER_LINEAR>::build(readDFArray[0], resParams[0]);

    constexpr auto resizeDFArray = ResizeRead<INTER_LINEAR>::build(readDFArray, resParams);
    using ReadArrayTye = decltype(readDFArray);
    const auto resizeDFArray2 = ResizeRead<INTER_LINEAR, PRESERVE_AR>::build(readDFArray, resParams, defaultArray);

    return 0;
}
