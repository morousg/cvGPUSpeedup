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

#include <fused_kernel/algorithms/image_processing/warping.cuh>
#include <fused_kernel/core/execution_model/memory_operations.cuh>

int launch() {
    constexpr auto test = fk::WarpingParameters<fk::Affine>{};
    static_assert(test.transformMatrix.dims.height == 2, "Height is not == 2");
    static_assert(test.transformMatrix.dims.width == 3, "Width is not == 3");

    constexpr auto test2 = fk::WarpingParameters<fk::Perspective>{};
    static_assert(test2.transformMatrix.dims.height == 3, "Height is not == 3");
    static_assert(test2.transformMatrix.dims.width == 3, "Width is not == 3");

    constexpr auto test3 = fk::PerThreadRead<fk::_2D, uchar3>::build({ nullptr, {16,16,16*3} });
    static_assert(test3.getActiveThreads().x == 16 && test3.getActiveThreads().y == 16 && test3.getActiveThreads().z == 1,
        "Incorrect active threads values");

    constexpr auto test4 = fk::Warping<fk::Affine, void>::build(fk::WarpingParameters<fk::Affine>{ {}, {8,8}});
    constexpr auto test4_2 = fk::Warping<fk::Perspective, void>::build(fk::WarpingParameters<fk::Perspective>{ {}, { 8,8 }});

    constexpr auto test5 = test3.then(test4);
    
    constexpr auto actThrWarp = test5.getActiveThreads();
    static_assert(actThrWarp.x == 8 && actThrWarp.y == 8 && actThrWarp.z == 1,
                  "Incorrect active threads for test5");

    constexpr auto actThrPTR = test5.back_function.getActiveThreads();
    static_assert(actThrPTR.x == 16 && actThrPTR.y == 16 && actThrPTR.z == 1,
                  "Incorrect active threads for test3 when fused with test5");

    using AffineType = float[2][3];
    using PerspectiveType = float[3][3];
    static_assert(!std::is_same_v<AffineType, PerspectiveType>,
                  "We can't differentiate the types float[2][3] and float[3][3]");
    static_assert(std::is_same_v<AffineType, decltype(test4.params.transformMatrix.data)>,
                  "Wrong dimensions for transform matrix");
    static_assert(std::is_same_v<PerspectiveType, decltype(test4_2.params.transformMatrix.data)>,
                  "Wrong dimensions for transform matrix");

    return 0;
}
