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

#include "tests/main.h"

#include <fused_kernel/algorithms/image_processing/crop.cuh>
#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>

using namespace fk;

int launch() {
    constexpr RawPtr<_2D, uchar3> input{ nullptr, {128, 128, 128*sizeof(uchar3)}};
    constexpr auto readIOp = PerThreadRead<_2D, uchar3>::build(input);
    using ReadIOp = decltype(readIOp);

    constexpr Rect aCrop(10, 12, 20, 30);
    constexpr auto cropOp = Crop<ReadIOp>::build(readIOp, aCrop);
    static_assert(cropOp.getActiveThreads().x == 20, "Wrong x");
    static_assert(cropOp.getActiveThreads().y == 30, "Wrong y");
    constexpr auto fusedCrop = readIOp.then(Crop<>::build(Rect(11, 9, 10, 10)));
    static_assert(fusedCrop.getActiveThreads().x == 10, "Wrong x");

    constexpr std::array<Rect, 2> rects{ aCrop, Rect(15,15, 50, 20)};

    constexpr auto batchCrop = readIOp.then(Crop<>::build(rects));
    using BatchedCrop = decltype(batchCrop);

    constexpr auto batchCropResize = 
        readIOp.then(Crop<>::build(rects))
               .then(ResizeRead<INTER_LINEAR>::build(Size(100, 100)));

    static_assert(batchCropResize.getActiveThreads().x == 100);
    static_assert(batchCropResize.getActiveThreads().y == 100);
    static_assert(batchCropResize.getActiveThreads().z == 2);

    return 0;
}