/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>

namespace fk {

    struct M3x3Float {
        const float3 x;
        const float3 y;
        const float3 z;
    };

    struct MxVFloat3 {
        BINARY_DECL_EXEC(float3, float3, M3x3Float) {
            const float3 xOut = input * params.x;
            const float3 yOut = input * params.y;
            const float3 zOut = input * params.z;
            using Reduce = VectorReduce<float3, Sum<float>>;
            return { Reduce::exec(xOut), Reduce::exec(yOut), Reduce::exec(zOut) };
        }
    };

} //namespace fk
