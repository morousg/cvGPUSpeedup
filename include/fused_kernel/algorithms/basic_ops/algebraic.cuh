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

#ifndef FK_ALGEBRAIC
#define FK_ALGEBRAIC

#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.cuh>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {

    struct M3x3Float {
        const float3 x;
        const float3 y;
        const float3 z;
    };

    struct MxVFloat3 {
        using OutputType = float3;
        using InputType = float3;
        using ParamsType = M3x3Float; 
        using InstanceType = BinaryType; 
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            const float3 xOut = input * params.x;
            const float3 yOut = input * params.y;
            const float3 zOut = input * params.z;
            using Reduce = VectorReduce<float3, Add<float>>;
            return { Reduce::exec(xOut), Reduce::exec(yOut), Reduce::exec(zOut) };
        }
        using InstantiableType = Binary<MxVFloat3>;
        DEFAULT_BINARY_BUILD
    };
} //namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
