/* Copyright 2023-2024 Oscar Amoros Huguet

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

#include <fused_kernel/algorithms/image_processing/interpolation.cuh>

namespace fk {
    struct ComputeResizePoint {
        using ParamsType = float2;
        using InputType = Point;
        using InstanceType = BinaryType;
        using OutputType = float2;
        static __device__ __forceinline__ const OutputType exec(const InputType& thread, const ParamsType& params) {
            // This is what makes the interpolation a resize operation
            const float fx = params.x;
            const float fy = params.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;

            return { src_x, src_y };
        }
    };

    template <InterpolationType IType>
    struct ResizeReadParams {
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
    };

    template <typename BackFunction_, InterpolationType IType>
    struct ResizeRead {
        using BackFunction = BackFunction_;
        using ParamsType = ResizeReadParams<IType>;
        static constexpr bool THREAD_FUSION{ false };
        using InstanceType = ReadBackType;
        using OutputType = typename Interpolate<BackFunction, IType>::OutputType;
        using ReadDataType = typename BackFunction::Operation::OutputType;
        static __device__ __forceinline__ const OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function) {
            const float fx = params.src_conv_factors.x;
            const float fy = params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of ResizeRead, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the ResizeRead implementation, and not a template variable.
            return Interpolate<BackFunction, IType>::exec(rezisePoint, params.params, back_function);
        }
    };
}; // namespace fk


